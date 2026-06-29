import os
import signal
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as app_module
import db as db_module
from app import app
from db import get_db
from models import Base, DetectionObject, PredictionSession


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    test_engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(
        bind=test_engine,
        autoflush=False,
        autocommit=False,
    )

    Base.metadata.create_all(bind=test_engine)

    app_module.UPLOAD_DIR = str(tmp_path / "original")
    app_module.PREDICTED_DIR = str(tmp_path / "predicted")
    os.makedirs(app_module.UPLOAD_DIR, exist_ok=True)
    os.makedirs(app_module.PREDICTED_DIR, exist_ok=True)

    db_module.engine = test_engine
    db_module.SessionLocal = TestingSessionLocal

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    yield TestingSessionLocal

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture
def client():
    return TestClient(app)


def seed_prediction(session_factory, uid, original_image, predicted_image):
    session = session_factory()
    try:
        session.add(
            PredictionSession(
                uid=uid,
                original_image=original_image,
                predicted_image=predicted_image,
            )
        )
        session.commit()
    finally:
        session.close()


def seed_detection(session_factory, prediction_uid, label, score, box):
    session = session_factory()
    try:
        session.add(
            DetectionObject(
                prediction_uid=prediction_uid,
                label=label,
                score=score,
                box=str(box),
            )
        )
        session.commit()
    finally:
        session.close()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestPredictionsByLabel(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_empty_label_returns_400(self):
        response = self.client.get("/predictions/label/ ")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Label cannot be empty")

    def test_no_match_returns_empty_list(self):
        response = self.client.get("/predictions/label/person")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_match_returns_sessions_with_label(self):
        seed_prediction(db_module.SessionLocal, "uid-1", "orig1.jpg", "pred1.jpg")
        seed_detection(db_module.SessionLocal, "uid-1", "person", 0.91, [10, 20, 100, 200])
        seed_detection(db_module.SessionLocal, "uid-1", "person", 0.85, [30, 40, 120, 220])
        seed_detection(db_module.SessionLocal, "uid-1", "car", 0.70, [0, 0, 50, 50])

        seed_prediction(db_module.SessionLocal, "uid-2", "orig2.jpg", "pred2.jpg")
        seed_detection(db_module.SessionLocal, "uid-2", "person", 0.77, [5, 5, 60, 90])

        response = self.client.get("/predictions/label/person")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data), 2)
        self.assertEqual({s["uid"] for s in data}, {"uid-1", "uid-2"})

        session1 = next(s for s in data if s["uid"] == "uid-1")
        self.assertEqual(len(session1["detection_objects"]), 2)
        self.assertTrue(all(o["label"] == "person" for o in session1["detection_objects"]))


class TestPredictionsByScore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_out_of_range_returns_400(self):
        response = self.client.get("/predictions/score/1.5")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "min_score must be between 0.0 and 1.0")

    def test_no_match_returns_empty_list(self):
        response = self.client.get("/predictions/score/0.9")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_returns_objects_at_or_above_threshold(self):
        seed_prediction(db_module.SessionLocal, "uid-1", "o.jpg", "p.jpg")
        seed_detection(db_module.SessionLocal, "uid-1", "person", 0.91, [10, 20, 100, 200])
        seed_detection(db_module.SessionLocal, "uid-1", "dog", 0.75, [1, 2, 3, 4])
        seed_detection(db_module.SessionLocal, "uid-1", "car", 0.40, [0, 0, 50, 50])

        response = self.client.get("/predictions/score/0.75")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data), 2)
        self.assertTrue(all(o["score"] >= 0.75 for o in data))
        self.assertEqual(data[0]["prediction_uid"], "uid-1")


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("app.s3")
    @patch("app.Image")
    @patch("app.model")
    def test_no_detections(self, mock_model, mock_image, mock_s3):
        fake_result = MagicMock()
        fake_result.boxes = []
        fake_result.plot.return_value = MagicMock()
        mock_model.return_value = [fake_result]
        mock_model.names = {}
        mock_image.fromarray.return_value.save.return_value = None

        response = self.client.post(
            "/predict",
            json={
                "image_s3_key": "chat-1/uid-1/original/image.jpg",
                "prediction_id": "uid-1",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detection_count"], 0)
        self.assertEqual(body["labels"], [])
        self.assertEqual(body["prediction_uid"], "uid-1")
        mock_s3.download_file.assert_called_once()
        mock_s3.upload_file.assert_called_once()

    @patch("app.s3")
    @patch("app.Image")
    @patch("app.model")
    def test_with_detections(self, mock_model, mock_image, mock_s3):
        fake_box = MagicMock()
        fake_box.cls[0].item.return_value = 0
        fake_box.conf = [0.88]
        fake_box.xyxy[0].tolist.return_value = [1, 2, 3, 4]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        fake_result.plot.return_value = MagicMock()
        mock_model.return_value = [fake_result]
        mock_model.names = {0: "person"}
        mock_image.fromarray.return_value.save.return_value = None

        response = self.client.post(
            "/predict",
            json={
                "image_s3_key": "chat-1/uid-2/original/image.jpg",
                "prediction_id": "uid-2",
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detection_count"], 1)
        self.assertEqual(body["labels"], ["person"])
        # predicted key is derived by swapping the original/ segment
        mock_s3.upload_file.assert_called_once()
        self.assertEqual(
            mock_s3.upload_file.call_args[0][2],
            "chat-1/uid-2/predicted/image.jpg",
        )


class TestGetPredictionByUid(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_found_returns_session_with_objects(self):
        seed_prediction(db_module.SessionLocal, "uid-1", "orig.jpg", "pred.jpg")
        seed_detection(db_module.SessionLocal, "uid-1", "person", 0.9, [1, 2, 3, 4])

        response = self.client.get("/prediction/uid-1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], "uid-1")
        self.assertEqual(len(data["detection_objects"]), 1)

    def test_unknown_uid_returns_404(self):
        response = self.client.get("/prediction/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")


class TestReady(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        app_module.is_shutting_down = False

    def test_ready_when_running(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_ready_when_shutting_down_returns_503(self):
        app_module.is_shutting_down = True
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Service is shutting down")


class TestSigtermHandler(unittest.TestCase):
    def tearDown(self):
        app_module.is_shutting_down = False

    def test_handle_sigterm_sets_flag_and_exits(self):
        with self.assertRaises(SystemExit) as cm:
            app_module.handle_sigterm(signal.SIGTERM, None)
        self.assertEqual(cm.exception.code, 0)
        self.assertTrue(app_module.is_shutting_down)
