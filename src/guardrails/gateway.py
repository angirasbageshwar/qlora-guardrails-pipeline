"""STEP 6 — Guardrail Gateway: the ONLY door to the model.

  client ──► gateway :9000 ──► vLLM :8000 (OpenAI-compatible)
              │ input rails: jailbreak patterns + PII scrub (+ optional NeMo)
              │ output rails: PII scrub before the reply leaves
              └ every block/mask decision is logged for audit

Run:  uvicorn src.guardrails.gateway:app --host 0.0.0.0 --port 9000
Then call it exactly like OpenAI:  POST /v1/chat/completions
Streaming: input rails always apply; the output PII scrub applies to
non-streaming responses (documented trade-off — see GUIDE §guardrails).
"""
import logging
import os
import time
from collections import defaultdict, deque

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.guardrails.jailbreak_filter import check as jailbreak_check
from src.guardrails.pii_filter import scrub

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("guardrail-gateway")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
BLOCK_ON_PII = os.getenv("BLOCK_ON_PII", "false").lower() == "true"
USE_NEMO = os.getenv("USE_NEMO", "false").lower() == "true"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "0"))                  # 0 = no cap
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "0"))  # per client IP; 0 = off
_UI_PATH = os.path.join(os.path.dirname(__file__), "ui", "index.html")
_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="LLM Guardrail Gateway", version="1.0.0")

_nemo_rails = None
if USE_NEMO:
    try:
        from nemoguardrails import LLMRails, RailsConfig
        _cfg = RailsConfig.from_path(
            os.path.join(os.path.dirname(__file__), "nemo_config"))
        _nemo_rails = LLMRails(_cfg)
        log.info("NeMo Guardrails rails loaded")
    except Exception as exc:  # noqa: BLE001 — keep the deterministic layer alive regardless
        log.warning("NeMo unavailable (%s) — continuing with regex rails only", exc)


def _extract_user_text(body: dict) -> str:
    return "\n".join(
        str(m.get("content", "")) for m in body.get("messages", [])
        if m.get("role") == "user")


def apply_input_rails(body: dict) -> dict:
    """Mutates/validates the request. Raises HTTPException(400) on block."""
    user_text = _extract_user_text(body)

    jb = jailbreak_check(user_text)
    if jb.blocked:
        log.warning("BLOCKED jailbreak attempt: %s", jb.matched)
        raise HTTPException(status_code=400, detail={
            "error": "request_blocked", "reason": "jailbreak_pattern",
            "matched_rules": jb.matched})

    pii = scrub(user_text)
    if pii.has_pii:
        log.warning("PII in prompt: %s", pii.found)
        if BLOCK_ON_PII:
            raise HTTPException(status_code=400, detail={
                "error": "request_blocked", "reason": "pii_detected",
                "found": pii.found})
        for m in body.get("messages", []):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = scrub(m["content"]).clean_text
    return body


def _client_ip(request: Request) -> str:
    # Set by Cloudflare / Vercel in front of us; falls back to the socket peer.
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _rate_limited(ip: str) -> bool:
    if RATE_LIMIT_PER_MIN <= 0:
        return False
    now, window = time.monotonic(), _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        return True
    window.append(now)
    return False


def apply_output_rails(text: str) -> tuple[str, dict]:
    result = scrub(text)
    if result.has_pii:
        log.warning("PII in model output masked: %s", result.found)
    return result.clean_text, result.found


@app.get("/", include_in_schema=False)
async def ui() -> FileResponse:
    return FileResponse(_UI_PATH)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "backend": VLLM_BASE_URL,
            "nemo": bool(_nemo_rails), "block_on_pii": BLOCK_ON_PII}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness: can we reach the model? (/health stays liveness-only for CI.)"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
    except (httpx.HTTPError, ValueError) as exc:
        return JSONResponse(status_code=503, content={
            "ready": False, "error": type(exc).__name__})
    return JSONResponse(content={"ready": True, "models": models})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """OpenAI-compatible model list, passed through from the backend."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail={"error": "backend_unavailable"}) from exc


@app.post("/v1/chat/completions")
async def chat(request: Request):
    if _rate_limited(_client_ip(request)):
        raise HTTPException(status_code=429, detail={
            "error": "rate_limited", "limit_per_minute": RATE_LIMIT_PER_MIN})
    body = await request.json()
    t0 = time.perf_counter()
    input_pii = scrub(_extract_user_text(body)).found   # reported back to the client
    body = apply_input_rails(body)
    if MAX_TOKENS:
        body["max_tokens"] = min(int(body.get("max_tokens") or MAX_TOKENS), MAX_TOKENS)
    stream = bool(body.get("stream"))

    if stream:
        # The relay owns its client: it must outlive this handler's return.
        async def relay():
            async with httpx.AsyncClient(timeout=120.0) as client, client.stream(
                "POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=body
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(relay(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
        except httpx.RequestError as exc:
            log.error("Model backend unreachable at %s: %s", VLLM_BASE_URL, exc)
            raise HTTPException(status_code=502, detail={
                "error": "backend_unavailable"}) from exc
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        data = resp.json()

    masked_total: dict[str, int] = {}
    for choice in data.get("choices", []):
        content = choice.get("message", {}).get("content")
        if isinstance(content, str):
            clean, found = apply_output_rails(content)
            choice["message"]["content"] = clean
            for k, v in found.items():
                masked_total[k] = masked_total.get(k, 0) + v

    user_turns = [m.get("content") for m in body.get("messages", []) if m.get("role") == "user"]
    data["guardrails"] = {
        "input_pii_policy": "block" if BLOCK_ON_PII else "mask",
        "input_pii_masked": input_pii,
        # what the model actually saw, so clients can keep only masked history
        "forwarded_prompt": user_turns[-1] if input_pii and user_turns else None,
        "output_pii_masked": masked_total,
        "gateway_latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return JSONResponse(content=data)
