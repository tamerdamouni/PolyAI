import os
import tempfile
import unittest
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


