import os
import signal
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

import app as app_module
from app import app, init_db, save_prediction_session, save_detection_object

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr("app.DB_PATH", db_file)
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestPredictionsByLabel(unittest.TestCase):
    def setUp(self):
        # isolated DB in a temp directory before each test
        self.tmp_dir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = os.path.join(self.tmp_dir.name, "test.db")
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        # Discard the temp DB after each test
        self.tmp_dir.cleanup()

    def test_empty_label_returns_400(self):
        response = self.client.get("/predictions/label/ ")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Label cannot be empty")

    def test_no_match_returns_empty_list(self):
        response = self.client.get("/predictions/label/person")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_match_returns_sessions_with_label(self):
        # Session with two "person" objects and one "car"
        save_prediction_session("uid-1", "orig1.jpg", "pred1.jpg")
        save_detection_object("uid-1", "person", 0.91, [10, 20, 100, 200])
        save_detection_object("uid-1", "person", 0.85, [30, 40, 120, 220])
        save_detection_object("uid-1", "car", 0.70, [0, 0, 50, 50])

        # Another session that also has a "person"
        save_prediction_session("uid-2", "orig2.jpg", "pred2.jpg")
        save_detection_object("uid-2", "person", 0.77, [5, 5, 60, 90])

        response = self.client.get("/predictions/label/person")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Both sessions are returned
        self.assertEqual(len(data), 2)
        self.assertEqual({s["uid"] for s in data}, {"uid-1", "uid-2"})

        # uid-1 returns its two person objects, the car is excluded
        session1 = next(s for s in data if s["uid"] == "uid-1")
        self.assertEqual(len(session1["detection_objects"]), 2)
        self.assertTrue(all(o["label"] == "person" for o in session1["detection_objects"]))


class TestPredictionsByScore(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = os.path.join(self.tmp_dir.name, "test.db")
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_out_of_range_returns_400(self):
        response = self.client.get("/predictions/score/1.5")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "min_score must be between 0.0 and 1.0")

    def test_no_match_returns_empty_list(self):
        response = self.client.get("/predictions/score/0.9")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_returns_objects_at_or_above_threshold(self):
        save_prediction_session("uid-1", "o.jpg", "p.jpg")
        save_detection_object("uid-1", "person", 0.91, [10, 20, 100, 200])
        save_detection_object("uid-1", "dog", 0.75, [1, 2, 3, 4])
        save_detection_object("uid-1", "car", 0.40, [0, 0, 50, 50])

        response = self.client.get("/predictions/score/0.75")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(len(data), 2)
        self.assertTrue(all(o["score"] >= 0.75 for o in data))
        self.assertEqual(data[0]["prediction_uid"], "uid-1")


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = os.path.join(self.tmp_dir.name, "test.db")
        # Keep uploaded/predicted files inside the temp dir, not the repo
        app_module.UPLOAD_DIR = os.path.join(self.tmp_dir.name, "original")
        app_module.PREDICTED_DIR = os.path.join(self.tmp_dir.name, "predicted")
        os.makedirs(app_module.UPLOAD_DIR, exist_ok=True)
        os.makedirs(app_module.PREDICTED_DIR, exist_ok=True)
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_invalid_extension_returns_400(self):
        response = self.client.post(
            "/predict",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Only image files are supported")

    @patch("app.Image")
    @patch("app.model")
    def test_no_detections(self, mock_model, mock_image):
        fake_result = MagicMock()
        fake_result.boxes = []
        mock_model.return_value = [fake_result]
        mock_model.names = {}

        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", b"fake image bytes", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detection_count"], 0)
        self.assertEqual(body["labels"], [])
        self.assertIn("prediction_uid", body)

    @patch("app.Image")
    @patch("app.model")
    def test_with_detections(self, mock_model, mock_image):
        # One fake detected box labelled "person"
        fake_box = MagicMock()
        fake_box.cls[0].item.return_value = 0
        fake_box.conf = [0.88]
        fake_box.xyxy[0].tolist.return_value = [1, 2, 3, 4]

        fake_result = MagicMock()
        fake_result.boxes = [fake_box]
        mock_model.return_value = [fake_result]
        mock_model.names = {0: "person"}

        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", b"fake image bytes", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["detection_count"], 1)
        self.assertEqual(body["labels"], ["person"])


class TestGetPredictionByUid(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = os.path.join(self.tmp_dir.name, "test.db")
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_found_returns_session_with_objects(self):
        save_prediction_session("uid-1", "orig.jpg", "pred.jpg")
        save_detection_object("uid-1", "person", 0.9, [1, 2, 3, 4])

        response = self.client.get("/prediction/uid-1")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], "uid-1")
        self.assertEqual(len(data["detection_objects"]), 1)

    def test_unknown_uid_returns_404(self):
        response = self.client.get("/prediction/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")


class TestGetPredictionImage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        app_module.DB_PATH = os.path.join(self.tmp_dir.name, "test.db")
        init_db()
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_found_returns_image_file(self):
        image_path = os.path.join(self.tmp_dir.name, "pred.jpg")
        with open(image_path, "wb") as f:
            f.write(b"fake image content")
        save_prediction_session("uid-1", "orig.jpg", image_path)

        response = self.client.get("/prediction/uid-1/image")
        self.assertEqual(response.status_code, 200)

    def test_missing_file_returns_404(self):
        # Session exists but the predicted image file is gone
        save_prediction_session("uid-1", "orig.jpg", "/no/such/file.jpg")
        response = self.client.get("/prediction/uid-1/image")
        self.assertEqual(response.status_code, 404)

    def test_unknown_uid_returns_404(self):
        response = self.client.get("/prediction/nope/image")
        self.assertEqual(response.status_code, 404)


class TestReady(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        # Reset the module-level flag so other tests are unaffected
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


