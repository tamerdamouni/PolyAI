from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
import logging
import os
import time
import uuid
import shutil
import signal
import sys
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import DetectionObject, PredictionSession


class PredictionResponse(BaseModel):
    prediction_uid: str
    detection_count: int
    labels: list[str]
    time_took: str  # wall-clock seconds, formatted "1.23"

is_shutting_down = False

# Graceful shutdown: handle SIGTERM so systemd stops the service cleanly
def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    # Perform cleanup: close DB connections, finish pending work, etc.
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)")
else:  # pragma: no cover
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  


def format_timestamp(timestamp):
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Predict objects in an image
    """
    start = time.perf_counter()
    filename = file.filename.lower()
    if not(filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png")):
        raise HTTPException(status_code=400, detail="Only image files are supported")
    
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    db.add(
        PredictionSession(
            uid=uid,
            original_image=original_path,
            predicted_image=predicted_path,
        )
    )

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        db.add(
            DetectionObject(
                prediction_uid=uid,
                label=label,
                score=score,
                box=str(bbox),
            )
        )
        detected_labels.append(label)

    db.commit()

    return PredictionResponse(
        prediction_uid=uid,
        detection_count=len(results[0].boxes),
        labels=detected_labels,
        time_took=f"{time.perf_counter() - start:.2f}",
    )

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = db.query(DetectionObject).filter_by(prediction_uid=uid).all()

    return {
        "uid": session.uid,
        "timestamp": format_timestamp(session.timestamp),
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box,
            }
            for obj in objects
        ],
    }


@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):
    """
    Get all prediction sessions that contain detected object with label 
    """
    if not label.strip():
        raise HTTPException(status_code=400, detail="Label cannot be empty")

    rows = (
        db.query(DetectionObject, PredictionSession)
        .join(PredictionSession, DetectionObject.prediction_uid == PredictionSession.uid)
        .filter(DetectionObject.label == label)
        .all()
    )

    # Group the matching objects under their session (one row per object)
    sessions = {}
    for obj, session in rows:
        uid = session.uid
        if uid not in sessions:
            sessions[uid] = {
                "uid": uid,
                "timestamp": format_timestamp(session.timestamp),
                "detection_objects": [],
            }
        sessions[uid]["detection_objects"].append({
            "id": obj.id,
            "label": obj.label,
            "score": obj.score,
            "box": obj.box,
        })

    return list(sessions.values())


@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float, db: Session = Depends(get_db)):
    """
    Get all detection objects with a confidence score >= min_score
    """
    if not 0.0 <= min_score <= 1.0:
        raise HTTPException(status_code=400, detail="min_score must be between 0.0 and 1.0")

    rows = db.query(DetectionObject).filter(DetectionObject.score >= min_score).all()

    return [
        {
            "id": row.id,
            "prediction_uid": row.prediction_uid,
            "label": row.label,
            "score": row.score,
            "box": row.box,
        }
        for row in rows
    ]


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session or not os.path.exists(session.predicted_image):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(session.predicted_image)


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}

@app.get("/health2")
def health2():
    return {"status": "fine"}

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
