"""FastAPI app wiring Copilot onto the OpenAI Chat Completions API."""

import json
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from copilot import CopilotClient
from copilot.driver import ClearanceRequired

from .config import MODEL_NAME, RATE_LIMIT_BURST, RATE_LIMIT_RPM
from .openai_format import (
    completion_response,
    new_id,
    sse_event,
    stream_chunk,
)
from .prompt import messages_to_prompt
from .ratelimit import TokenBucket
from .schemas import ChatCompletionRequest

app = FastAPI(title="Copilot OpenAI-compatible API", version="1.0.0")
# Server runs headless and must never pop a visible browser mid-request. With
# both recovery passes disabled, an expired clearance surfaces immediately as a
# 503 (see ClearanceRequired handling below) so an operator can re-clear out of
# band (`python -m copilot login`). Headless auto-solve is intentionally off:
# it's unreliable on low-trust egress and a failed pass can wedge the session.
client = CopilotClient(anonymous=False, interactive_clear=False, headless_clear=False)

_CLEARANCE_HELP = (
    "Cloudflare clearance expired and could not be refreshed headlessly. "
    "Re-clear in a browser: run `python -m copilot login` (or `python tests/diagnostic.py`) "
    "and pass the 'verify you're human' check, then retry."
)

# Self-imposed rate limit on top of the concurrency lock below: this caps
# requests-per-minute, the lock caps requests-in-flight. See server/ratelimit.py.
_rate_limiter = TokenBucket(RATE_LIMIT_RPM, RATE_LIMIT_BURST)


def _rate_limited_response():
    """Spend a token; return an OpenAI-shaped 429 if none left, else ``None``."""
    allowed, wait = _rate_limiter.try_acquire()
    if allowed:
        return None
    secs = max(1, round(wait))
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": str(secs)},
        content={"error": {
            "message": (
                f"Rate limit exceeded (>{RATE_LIMIT_RPM:g} req/min). "
                f"Retry in {secs}s."
            ),
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
        }},
    )

# Copilot's per-account chat socket doesn't tolerate concurrent conversations
# from one process (parallel requests error out or hang). This server bridges a
# single signed-in account, so we serialize upstream calls: concurrent HTTP
# requests queue here and run one at a time. Predictable, at the cost of
# parallelism — fine for a personal bridge.
_upstream_lock = threading.Lock()


def _stream(prompt: str, model: str, conversation_id=None):
    """Yield OpenAI ``chat.completion.chunk`` SSE events for ``prompt``.

    ``conversation_id`` continues an existing Copilot thread; ``None`` starts a
    fresh one (its id is emitted on the final chunk).
    """
    cid = new_id()
    created = int(time.time())
    try:
        with _upstream_lock:  # one upstream chat at a time (released on disconnect)
            yield sse_event(stream_chunk(cid, created, model, {"role": "assistant"}))
            stream = client.stream(prompt, conversation_id=conversation_id)
            for piece in stream:
                if isinstance(piece, str) and piece:
                    yield sse_event(stream_chunk(cid, created, model, {"content": piece}))
            # Copilot's conversation id is known once the stream has run; emit it
            # on the final chunk so callers can track the upstream thread.
            yield sse_event(
                stream_chunk(
                    cid, created, model, {}, finish="stop",
                    conversation_id=stream.conversation_id,
                )
            )
    except ClearanceRequired:
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {_CLEARANCE_HELP}]"}, finish="error")
        )
    except Exception as exc:  # surface errors to the client instead of hanging
        yield sse_event(
            stream_chunk(cid, created, model, {"content": f"\n[error: {exc}]"}, finish="error")
        )
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "microsoft"}
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    prompt = messages_to_prompt(req.messages)
    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content in messages", "type": "invalid_request_error"}},
        )
    model = req.model or MODEL_NAME

    # Enforce the per-minute ceiling before touching the upstream lock, so excess
    # callers get a fast 429 instead of piling up behind the serialized queue.
    limited = _rate_limited_response()
    if limited is not None:
        return limited

    if req.stream:
        return StreamingResponse(
            _stream(prompt, model, req.conversation_id), media_type="text/event-stream"
        )

    try:
        with _upstream_lock:  # serialize: one upstream chat at a time
            reply = client.chat(prompt, conversation_id=req.conversation_id)
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )
    return completion_response(reply.text, model, reply.conversation_id)


import logging
logger = logging.getLogger("uvicorn")

@app.post("/v1/responses")
def responses(req: dict):
    """Minimal /v1/responses compatibility — translates to chat completions."""
    logger.info("responses request: stream=%s", req.get("stream"))
    model = req.get("model") or MODEL_NAME
    inp = req.get("input", "")
    if isinstance(inp, list):
        texts = []
        for msg in inp:
            content = msg.get("content", "")
            if isinstance(content, list):
                texts.append(" ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                ))
            elif isinstance(content, str):
                texts.append(content)
        prompt = "\n".join(texts)
    else:
        prompt = str(inp)
    if not prompt.strip():
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "no text content", "type": "invalid_request_error"}},
        )
    limited = _rate_limited_response()
    if limited is not None:
        return limited
    try:
        if req.get("stream"):
            return StreamingResponse(
                _responses_stream(prompt, model, req.get("conversation_id")),
                media_type="text/event-stream",
            )
        with _upstream_lock:
            reply = client.chat(prompt, conversation_id=req.get("conversation_id"))
    except ClearanceRequired:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": _CLEARANCE_HELP, "type": "clearance_required"}},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )
    cid = f"resp_{reply.conversation_id or new_id()}"
    created = int(time.time())
    return {
        "id": cid,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "model": model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{new_id()}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply.text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def _responses_stream(prompt: str, model: str, conversation_id=None):
    """Stream /v1/responses as SSE events."""
    cid = f"resp_{new_id()}"
    created = int(time.time())
    try:
        with _upstream_lock:
            yield f"event: response.created\ndata: {json.dumps({'id': cid, 'object': 'response', 'created_at': created, 'model': model, 'status': 'in_progress'})}\n\n"
            stream = client.stream(prompt, conversation_id=conversation_id)
            msg_id = f"msg_{new_id()}"
            yield f"event: response.output_item.added\ndata: {json.dumps({'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': []})}\n\n"
            for piece in stream:
                if isinstance(piece, str) and piece:
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'id': msg_id, 'delta': piece})}\n\n"
            yield f"event: response.completed\ndata: {json.dumps({'id': cid, 'object': 'response', 'created_at': created, 'status': 'completed', 'model': model, 'output': [{'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': stream.conversation_id or ''}]}]})}\n\n"
    except ClearanceRequired:
        yield f"event: error\ndata: {json.dumps({'message': _CLEARANCE_HELP})}\n\n"
    except Exception as exc:
        yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/")
def root():
    return {"service": "Copilot OpenAI-compatible API", "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/responses"]}
