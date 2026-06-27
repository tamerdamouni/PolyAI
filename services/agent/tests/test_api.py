"""API-layer tests for the agent service.

These exercise the FastAPI endpoints with TestClient. The agentic loop
(`run_agent`) is mocked to a pre-defined result, so no LLM or YOLO call is made.
"""
import os

# Must be set before importing app: the startup feature check validates MODEL,
# and constructing the OpenAI client needs a key. The key is a dummy — the LLM
# is mocked in every test, so no real request is ever made.
os.environ.setdefault("MODEL", "openai:gpt-5.4-mini")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

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
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["response"] == "Hello there."
    assert body["iterations"] == 1
    assert body["tools_called"] == []
    assert body["tokens_used"] == {"input": 10, "output": 5, "total": 15}
    assert body["prediction_id"] is None
    assert body["annotated_image"] is None
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
        app_module, "_fetch_annotated_image", return_value="ZmFrZS1pbWFnZQ=="
    ):
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "what's in this?", "image_base64": "eA=="}
                ]
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "uid-123"
    assert body["annotated_image"] == "ZmFrZS1pbWFnZQ=="
    assert body["tools_called"] == ["detect_objects"]


def test_chat_rate_limit_returns_429():
    class FakeRateLimit(Exception):
        status_code = 429

    with patch.object(app_module, "run_agent", side_effect=FakeRateLimit("rate limited")):
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 429
    assert "rate-limited" in response.json()["detail"].lower()


def test_chat_non_rate_limit_error_still_raises():
    with patch.object(app_module, "run_agent", side_effect=ValueError("boom")):
        with pytest.raises(ValueError):
            client.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
