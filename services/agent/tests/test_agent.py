"""Unit tests for the agentic loop (`run_agent`).

The LLM is mocked with a fake chat model that returns pre-defined
``langchain_core.messages.AIMessage`` responses, so the loop runs without
hitting any real API. The YOLO call inside the ``detect_objects`` tool is mocked
at the httpx layer.
"""
import os

# Must be set before importing app: the startup feature check validates MODEL,
# and constructing the OpenAI client needs a key. The key is a dummy — the LLM
# is mocked in every test, so no real request is ever made.
os.environ.setdefault("MODEL", "openai:gpt-5.4-mini")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

import app as app_module
from app import run_agent


class FakeChatModel:
    """Stand-in for ``llm_with_tools`` — returns queued AIMessages in order."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, messages):
        return self._responses.pop(0)


def _usage(input_tokens, output_tokens):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _tool_call(name="detect_objects"):
    return {"name": name, "args": {}, "id": "call_1", "type": "tool_call"}


def test_final_answer_without_tools():
    final = AIMessage(content="The answer is 42.", usage_metadata=_usage(10, 5))

    with patch.object(app_module, "llm_with_tools", FakeChatModel([final])):
        result = run_agent([HumanMessage(content="hi")])

    assert result.response == "The answer is 42."
    assert result.iterations == 1
    assert result.tools_called == []
    assert result.prediction_id is None
    assert result.tokens_used.input == 10
    assert result.tokens_used.output == 5
    assert result.tokens_used.total == 15


def test_executes_tool_then_answers():
    tool_request = AIMessage(
        content="",
        tool_calls=[_tool_call()],
        usage_metadata=_usage(20, 8),
    )
    final = AIMessage(content="I found a cat.", usage_metadata=_usage(30, 6))

    # Mock the YOLO /predict call that detect_objects makes via httpx.
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "prediction_uid": "uid-123",
        "detection_count": 1,
        "labels": ["cat"],
    }
    fake_response.raise_for_status.return_value = None
    fake_client = MagicMock()
    fake_client.__enter__.return_value.post.return_value = fake_response

    # base64 for "fakeimage" so detect_objects gets past its image check.
    token = app_module._current_image_b64.set("ZmFrZWltYWdl")
    try:
        with patch.object(
            app_module, "llm_with_tools", FakeChatModel([tool_request, final])
        ), patch.object(app_module.httpx, "Client", return_value=fake_client):
            result = run_agent([HumanMessage(content="what's in the image?")])
    finally:
        app_module._current_image_b64.reset(token)

    assert result.response == "I found a cat."
    assert result.iterations == 2
    assert result.tools_called == ["detect_objects"]
    assert result.prediction_id == "uid-123"
    # Tokens summed across both LLM calls: (20+8) + (30+6) = 64
    assert result.tokens_used.total == 64
    assert result.tokens_used.input == 50
    assert result.tokens_used.output == 14


def test_hits_iteration_cap():
    class AlwaysRequestsTool:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[_tool_call()],
                usage_metadata=_usage(1, 1),
            )

    # No image is set, so detect_objects returns an error string without calling
    # YOLO — the model never gets a final answer and we hit the cap.
    with patch.object(app_module, "llm_with_tools", AlwaysRequestsTool()):
        result = run_agent([HumanMessage(content="loop forever")], max_iterations=3)

    assert result.iterations == 3
    assert "couldn't complete" in result.response.lower()
    assert result.tools_called == ["detect_objects", "detect_objects", "detect_objects"]
