"""API-layer tests for the agent service.

These exercise the FastAPI endpoints with TestClient. The agentic loop
(`run_agent`) is mocked to a pre-defined result, so no LLM or YOLO call is made.
"""
import os

# Must be set before importing app: the startup check validates MODEL. boto3
# resolves credentials lazily, so constructing the Bedrock client needs no real
# AWS keys — the LLM is mocked in every test, so no request is ever made.
os.environ.setdefault("MODEL", "openai.gpt-oss-20b-1:0")
os.environ.setdefault("AWS_REGION", "us-east-1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import AgentResult, TokenUsage, app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_format():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "# HELP" in response.text
    assert "# TYPE" in response.text


def test_chat_returns_structured_response():
    fake_result = AgentResult(
        response="Hello there.",
        iterations=1,
        tools_called=[],
        tokens_used=TokenUsage(input=10, output=5, total=15),
        prediction_id=None,
    )
    with patch.object(app_module, "run_agent", return_value=fake_result):
        response = client.post(
            "/chat",
            json={"chat_id": "chat-1", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Hello there."
    assert body["iterations"] == 1
    assert body["tools_called"] == []
    assert body["tokens_used"] == {"input": 10, "output": 5, "total": 15}
    assert body["prediction_id"] is None
    assert body["annotated_image_url"] is None
    assert body["context_limit_exceeded"] is False
    assert isinstance(body["agent_loop_time_s"], (int, float))


def test_chat_includes_annotated_image_when_detection_happened():
    fake_result = AgentResult(
        response="I found a cat.",
        iterations=2,
        tools_called=["detect_objects"],
        tokens_used=TokenUsage(input=40, output=10, total=50),
        prediction_id="uid-123",
    )
    with patch.object(app_module, "run_agent", return_value=fake_result), patch.object(
        app_module, "_presign_predicted_url", return_value="https://s3.example/predicted.jpg"
    ), patch.object(app_module, "s3"):
        response = client.post(
            "/chat",
            json={
                "chat_id": "chat-1",
                "messages": [
                    {"role": "user", "content": "what's in this?", "image_base64": "eA=="}
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "uid-123"
    assert body["annotated_image_url"] == "https://s3.example/predicted.jpg"
    assert body["tools_called"] == ["detect_objects"]


def test_chat_uploads_image_up_front_when_present():
    fake_result = AgentResult(
        response="ok", iterations=1, tools_called=[], tokens_used=TokenUsage()
    )
    with patch.object(app_module, "run_agent", return_value=fake_result), patch.object(
        app_module, "s3"
    ) as fake_s3:
        client.post(
            "/chat",
            json={
                "chat_id": "chat-1",
                "messages": [
                    {"role": "user", "content": "blur it", "image_base64": "eA=="}
                ],
            },
        )
    fake_s3.put_object.assert_called_once()
    key = fake_s3.put_object.call_args.kwargs["Key"]
    assert key.startswith("chat-1/")
    assert key.endswith("/original/image.jpg")


def test_chat_includes_edited_image_when_transform_happened():
    fake_result = AgentResult(
        response="Blurred it.",
        iterations=2,
        tools_called=["blur"],
        tokens_used=TokenUsage(input=10, output=5, total=15),
        edited_image_key="chat-1/img/original/edited-x.png",
    )
    with patch.object(app_module, "run_agent", return_value=fake_result), patch.object(
        app_module, "_presign_get_url", return_value="https://s3.example/edited.png"
    ), patch.object(app_module, "s3"):
        response = client.post(
            "/chat",
            json={
                "chat_id": "chat-1",
                "messages": [
                    {"role": "user", "content": "blur it", "image_base64": "eA=="}
                ],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["edited_image_url"] == "https://s3.example/edited.png"
    assert body["tools_called"] == ["blur"]


def test_chat_rate_limit_returns_429():
    class FakeRateLimit(Exception):
        status_code = 429

    with patch.object(app_module, "run_agent", side_effect=FakeRateLimit("rate limited")):
        response = client.post(
            "/chat",
            json={"chat_id": "chat-1", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 429
    assert "rate-limited" in response.json()["detail"].lower()


def test_chat_non_rate_limit_error_still_raises():
    with patch.object(app_module, "run_agent", side_effect=ValueError("boom")):
        with pytest.raises(ValueError):
            client.post(
                "/chat",
                json={"chat_id": "chat-1", "messages": [{"role": "user", "content": "hello"}]},
            )
