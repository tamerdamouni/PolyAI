from fastapi import Depends, FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
import boto3
import logging
import os
import time
import signal
import sys
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_db
from models import DetectionObject, PredictionSession

load_dotenv()


class PredictRequest(BaseModel):
    image_s3_key: str
    prediction_id: str


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

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  


def format_timestamp(timestamp):
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict objects in an image stored in S3
    """
    start = time.perf_counter()
    image_s3_key = request.image_s3_key
    uid = request.prediction_id
    predicted_s3_key = image_s3_key.replace("/original/", "/predicted/")

    ext = os.path.splitext(image_s3_key)[1] or ".jpg"
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    s3.download_file(AWS_S3_BUCKET, image_s3_key, original_path)

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    s3.upload_file(predicted_path, AWS_S3_BUCKET, predicted_s3_key)

    db.add(
        PredictionSession(
            uid=uid,
            original_image=image_s3_key,
            predicted_image=predicted_s3_key,
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
