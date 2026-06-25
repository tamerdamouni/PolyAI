---
name: yolo-api-data-layer
description: Use when changing the YOLO service data layer (services/yolo/app.py) — refactoring SQLite to SQLAlchemy, adding/removing endpoints or queries, adding tables or columns, cascade deletes, making the DB backend configurable for Postgres, or writing tests that touch the database.
---

# YOLO API Data Layer (SQLAlchemy)

## Overview

The YOLO service (`services/yolo/app.py`) talks to its database with **SQLAlchemy ORM**, not raw `sqlite3`. The same code runs on SQLite in dev and Postgres in production, selected by the `DB_BACKEND` env var. Any work that touches how the service reads or writes prediction data goes through this layer.

**Core principle:** The HTTP contract is frozen. You may change *how* data is stored and queried; you may **never** change what a client sees — every path, status code, and JSON field stays byte-identical.

## When to Use

Trigger on prompts like:
- "refactor the api to use sqlalchemy"
- "add an endpoint GET /predictions/recent that returns the 10 most recent sessions"
- "add a UserFeedback table to track user ratings per prediction"
- "write tests for the /predict endpoint"
- "the database layer doesn't follow our architectural design, fix it"
- "delete a prediction session and all its detection objects by uid"
- "add a column `processing_time_ms` to the prediction_sessions table"
- "make the database backend configurable so we can use postgres in production"

## The Frozen Contract — verify BEFORE and AFTER any change

These responses must be **identical** to the pre-change behavior. After editing, re-read each endpoint and confirm:

| Method | Path | Success | Errors | Response shape (unchanged) |
|--------|------|---------|--------|----------------------------|
| POST | `/predict` | 200 | 400 `"Only image files are supported"` | `{prediction_uid, detection_count, labels}` |
| GET | `/prediction/{uid}` | 200 | 404 `"Prediction not found"` | `{uid, timestamp, original_image, predicted_image, detection_objects:[{id,label,score,box}]}` |
| GET | `/predictions/label/{label}` | 200 | 400 `"Label cannot be empty"` | `[{uid, timestamp, detection_objects:[{id,label,score,box}]}]` |
| GET | `/predictions/score/{min_score}` | 200 | 400 `"min_score must be between 0.0 and 1.0"` | `[{id, prediction_uid, label, score, box}]` |
| GET | `/prediction/{uid}/image` | 200 (FileResponse) | 404 `"Image not found"` | the image file |
| GET | `/health`, `/health2`, `/ready` | 200 / 503 | — | unchanged |

**`box` is a string.** It is stored with `str(bbox)` and returned as a string. Keep `box=str(bbox)` on insert and never convert it to a list/JSON. **`score` is a float, `id` is an int.** Validation order and error strings are part of the contract — keep the `400` checks (empty label, score range) exactly as written, before any DB access.

## Target File Layout

```
services/yolo/
  models.py   # declarative Base + ORM models (NEW)
  db.py       # engine, SessionLocal, get_db dependency (NEW)
  app.py      # endpoints use Depends(get_db) + ORM queries (REFACTORED)
```

### models.py

```python
from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)

    # cascade lets "delete a session and all its objects" be one db.delete(session)
    objects = relationship(
        "DetectionObject",
        backref="session",
        cascade="all, delete-orphan",
    )


class DetectionObject(Base):
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)  # stored as str(bbox) — keep it a string
```

The model maps 1:1 to the existing schema. **Do not rename columns or tables** — `prediction_sessions`, `detection_objects`, and every column name must match, or existing data and the frozen contract break.

### db.py

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "pass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "predictions")

if DB_BACKEND == "postgres":
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
else:
    DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# Declarative tables are created automatically — init_db() is gone.
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

`Base.metadata.create_all` replaces `init_db()`. Delete `init_db()` and its call in `if __name__ == "__main__"`. The Postgres `DB_NAME` defaults to `predictions` so it matches the course's docker command (`POSTGRES_DB=predictions`).

### app.py wiring

- Delete `import sqlite3`, `DB_PATH`, `init_db`, `save_prediction_session`, `save_detection_object`.
- Add `from fastapi import Depends`, `from sqlalchemy.orm import Session`, `from db import get_db`, and `from models import PredictionSession, DetectionObject`.
- Every endpoint that touches the DB gains `db: Session = Depends(get_db)` as its **last** parameter. Never call `get_db()` yourself — let FastAPI inject it.

## Raw SQL → ORM (per existing endpoint)

**INSERT** (`/predict`):
```python
db.add(PredictionSession(uid=uid, original_image=original_path, predicted_image=predicted_path))
db.add(DetectionObject(prediction_uid=uid, label=label, score=score, box=str(bbox)))
db.commit()
```

**SELECT one** (`/prediction/{uid}`):
```python
session = db.query(PredictionSession).filter_by(uid=uid).first()
if not session:
    raise HTTPException(status_code=404, detail="Prediction not found")
objects = db.query(DetectionObject).filter_by(prediction_uid=uid).all()
```
Build the response dict by hand from `session.uid`, `session.timestamp`, … and `[{ "id": o.id, "label": o.label, "score": o.score, "box": o.box } for o in objects]` — keep the exact keys above.

**JOIN / filter** (`/predictions/label/{label}`): query `DetectionObject` joined to `PredictionSession` filtered by `label`, then group objects by `uid` in Python exactly as today (`{uid, timestamp, detection_objects:[...]}`).

**Filter by value** (`/predictions/score/{min_score}`):
```python
rows = db.query(DetectionObject).filter(DetectionObject.score >= min_score).all()
```

**Single column** (`/prediction/{uid}/image`): fetch the session, use `session.predicted_image`, keep the `os.path.exists` + 404 check.

**DELETE with cascade** (new endpoints like "delete a session and its objects"):
```python
session = db.query(PredictionSession).filter_by(uid=uid).first()
if not session:
    raise HTTPException(status_code=404, detail="Prediction not found")
db.delete(session)   # cascade removes its detection_objects
db.commit()
```

## Adding tables, columns, and endpoints

- **New table** ("add a UserFeedback table"): add a new model class to `models.py` with `Base`. `create_all` picks it up — no migration step. Add the endpoint(s) with `Depends(get_db)`.
- **New column** ("add `processing_time_ms`"): add the `Column` to the model. SQLite has no auto-migrate, so on an existing dev DB note that the file must be recreated (delete `predictions.db`) or the column added manually; mention this rather than silently failing.
- **New query endpoint** ("GET /predictions/recent"): `db.query(PredictionSession).order_by(PredictionSession.timestamp.desc()).limit(10).all()`. Choose a response shape consistent with the table above and state it.

## Tests must keep passing — update them with the refactor

`tests/test_api.py` imports `init_db`, `save_prediction_session`, `save_detection_object` and monkeypatches `app.DB_PATH`. Those no longer exist, so the tests will fail unless you migrate them **in the same change**:

- Replace the `DB_PATH` monkeypatch / per-test temp DB with a temp SQLite engine bound to `SessionLocal`, and create tables with `Base.metadata.create_all(bind=test_engine)`.
- Inject the test session by overriding the dependency: `app.dependency_overrides[get_db] = lambda: test_session` (remember to clear it in teardown).
- Replace `save_prediction_session(...)` / `save_detection_object(...)` seed calls with `db.add(PredictionSession(...))` / `db.add(DetectionObject(...))` + `db.commit()` against the test session.
- Keep every assertion (status codes, field values, counts) exactly as-is — they encode the frozen contract.

Run `pytest tests/` from `services/yolo/`. **All tests must pass and coverage must not regress** versus before the change.

## requirements.txt

Add `sqlalchemy>=2.0` and, for the Postgres backend, `psycopg2-binary>=2.9`.

## Verification checklist (do not claim done until all pass)

- [ ] `grep -n "sqlite3\|DB_PATH\|init_db\|save_prediction_session\|save_detection_object\|execute(\"" app.py` returns nothing — no raw SQL, no `sqlite3`, no leftover helpers.
- [ ] Every DB endpoint has `db: Session = Depends(get_db)`; `get_db()` is never called directly.
- [ ] `models.py` and `db.py` exist; `init_db()` is deleted.
- [ ] `pytest tests/` — all green, coverage not lower than before.
- [ ] App boots: `python app.py` starts with no error and a request to each endpoint returns the same shape as before.
- [ ] Postgres smoke test (when relevant): with the course docker Postgres running, `export DB_BACKEND=postgres` and confirm identical behavior.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Partial refactor — some endpoints still use `sqlite3` | The grep check must be clean; convert **every** query. |
| Returning the ORM object / changing `timestamp` or `box` type | Build response dicts by hand; keep `box` a string, keep the same keys. |
| Removing `init_db()` but never creating tables | `Base.metadata.create_all(bind=engine)` in `db.py`. |
| Calling `get_db()` manually | Use `Depends(get_db)`; in tests use `app.dependency_overrides`. |
| Leaving `tests/test_api.py` importing the deleted helpers | Migrate the tests in the same change; all must pass. |
| Forgetting `sqlalchemy` (and `psycopg2-binary`) in requirements.txt | Add them. |
| Renaming a table or column | Keep `prediction_sessions` / `detection_objects` and all column names. |
