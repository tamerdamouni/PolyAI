import base64
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from typing import Optional

import boto3

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MODEL = os.environ.get("MODEL")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.environ.get("AWS_S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

PRESIGN_EXPIRY_SECONDS = 300

# Bedrock model IDs allowed by the account's IAM policy (bedrock_converse provider).
ALLOWED_MODELS = {
    "openai.gpt-oss-20b-1:0",
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported Bedrock model IDs:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_current_chat_id: ContextVar[Optional[str]] = ContextVar("current_chat_id", default=None)

@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    chat_id = _current_chat_id.get()
    prediction_id = str(uuid.uuid4())
    image_s3_key = f"{chat_id}/{prediction_id}/original/image.jpg"

    image_bytes = base64.b64decode(image_b64)
    s3.put_object(
        Bucket=AWS_S3_BUCKET,
        Key=image_s3_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            json={"image_s3_key": image_s3_key, "prediction_id": prediction_id},
        )
        response.raise_for_status()
    return json.dumps(response.json())


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects
}

# Client-side request throttle. LangChain's InMemoryRateLimiter only spaces out
# REQUESTS (it does not count tokens), so it keeps us under the per-minute REQUEST
# cap (RPM) but cannot enforce token-per-minute limits.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,    # ~30 requests/min
    check_every_n_seconds=0.1,  # how often to wake and check the token bucket
    max_bucket_size=1,          # no bursting — one in-flight request at a time
)

llm = init_chat_model(
    MODEL,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0,
    rate_limiter=rate_limiter,
)

# Verify the model supports the features this agent relies on, when its profile
# exposes them. Fail fast at startup if it doesn't.
REQUIRED_FEATURES = ("tool_calling", "structured_output")
_profile = getattr(llm, "profile", None) or {}
_missing = [f for f in REQUIRED_FEATURES if f in _profile and not _profile[f]]
if _missing:
    raise SystemExit(
        f"\n[ERROR] Model '{MODEL}' does not support required feature(s): "
        f"{', '.join(_missing)}.\nChoose a model whose profile supports: "
        f"{', '.join(REQUIRED_FEATURES)}.\n"
    )

llm_with_tools = llm.bind_tools(list(TOOLS.values()))


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class AgentResult(BaseModel):
    """Internal result carried out of the agent loop (not the HTTP response)."""
    response: str
    iterations: int
    tools_called: list[str]
    tokens_used: TokenUsage
    prediction_id: Optional[str] = None
    context_limit_exceeded: bool = False


def run_agent(history: list, max_iterations: int = 10) -> AgentResult:
    """
    Simple ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response (or we hit max_iterations).

    Returns an AgentResult carrying the final text plus loop metadata.
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    iterations = 0
    tools_called: list[str] = []
    prediction_id: Optional[str] = None
    tokens = TokenUsage()

    for _ in range(max_iterations):
        iterations += 1
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # Accumulate token usage across every LLM call in the loop
        usage = response.usage_metadata or {}
        tokens.input += usage.get("input_tokens", 0)
        tokens.output += usage.get("output_tokens", 0)
        tokens.total += usage.get("total_tokens", 0)

        # No tool calls, the model produced its final answer
        if not response.tool_calls:
            return AgentResult(
                response=response.text,
                iterations=iterations,
                tools_called=tools_called,
                tokens_used=tokens,
                prediction_id=prediction_id,
            )

        # Execute every tool the model requested
        for tool_call in response.tool_calls:
            tool_fn = TOOLS[tool_call["name"]]
            tool_result = tool_fn.invoke(tool_call)          # returns a ToolMessage
            messages.append(tool_result)
            tools_called.append(tool_call["name"])

            # Capture the prediction id from any detect_objects result
            try:
                payload = json.loads(tool_result.content)
                if payload.get("prediction_uid"):
                    prediction_id = payload["prediction_uid"]
            except (json.JSONDecodeError, AttributeError):
                pass

    # Hit the iteration cap without a final text answer
    return AgentResult(
        response="Sorry, I couldn't complete that within the allowed number of steps.",
        iterations=iterations,
        tools_called=tools_called,
        tokens_used=tokens,
        prediction_id=prediction_id,
    )


app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://dev.tamer.fursa.click:3000",
        "http://prod.tamer.fursa.click:3000",
        ],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first
    chat_id: str                        # stable id for the conversation, set by the client


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image_url: Optional[str] = None  # presigned S3 URL for the annotated image, or null
    agent_loop_time_s: float
    iterations: int
    tools_called: list[str]
    tokens_used: TokenUsage
    context_limit_exceeded: bool = False


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64          # saved for detect_objects tool
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    image_token = _current_image_b64.set(latest_image)
    chat_id_token = _current_chat_id.set(request.chat_id)
    try:
        start = time.perf_counter()
        result = run_agent(lc_messages)
        agent_loop_time_s = time.perf_counter() - start
    except Exception as exc:
        # If the provider's own retries are exhausted, a persistent 429 reaches
        # us here. Surface it as a clean 429 instead of an opaque 500.
        if _is_rate_limit_error(exc):
            raise HTTPException(
                status_code=429,
                detail="The model is rate-limited. Please try again shortly.",
            ) from exc
        raise
    finally:
        _current_image_b64.reset(image_token)
        _current_chat_id.reset(chat_id_token)

    annotated_image_url = None
    if result.prediction_id:
        annotated_image_url = _presign_predicted_url(request.chat_id, result.prediction_id)

    return ChatResponse(
        response=result.response,
        prediction_id=result.prediction_id,
        annotated_image_url=annotated_image_url,
        agent_loop_time_s=agent_loop_time_s,
        iterations=result.iterations,
        tools_called=result.tools_called,
        tokens_used=result.tokens_used,
        context_limit_exceeded=result.context_limit_exceeded,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a 429 across provider SDKs without importing each one."""
    # Anthropic / OpenAI expose .status_code; some wrap it in .response.status_code
    if getattr(exc, "status_code", None) == 429:
        return True
    if getattr(getattr(exc, "response", None), "status_code", None) == 429:
        return True
    # Google api_core raises ResourceExhausted with .code == 429
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "resourceexhausted" in text
        or "throttling" in text
    )


def _presign_predicted_url(chat_id: str, prediction_id: str) -> Optional[str]:
    """Return a short-lived presigned GET URL for the annotated image, or None on failure."""
    predicted_s3_key = f"{chat_id}/{prediction_id}/predicted/image.jpg"
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_S3_BUCKET, "Key": predicted_s3_key},
            ExpiresIn=PRESIGN_EXPIRY_SECONDS,
        )
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
