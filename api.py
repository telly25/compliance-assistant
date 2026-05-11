"""
API FastAPI — RegWatch

Endpoints :
  GET  /                  -> interface web
  POST /api/ask/stream    -> reponse en streaming (SSE)
  GET  /api/models        -> modele actif
"""

import json
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from query.rag import (
    MISTRAL_MODEL, LM_STUDIO_MODEL, LM_STUDIO_URL,
    LLM_MODE, N_RESULTS, SYSTEM_PROMPT, SOURCE_MAP, build_context
)
from ingest.embed import search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("regwatch")

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
_IS_PROD = os.environ.get("LLM_MODE") == "mistral"

app = FastAPI(
    title="RegWatch",
    # Désactive la doc OpenAPI en production (évite d'exposer la structure de l'API)
    docs_url=None if _IS_PROD else "/docs",
    redoc_url=None if _IS_PROD else "/redoc",
    openapi_url=None if _IS_PROD else "/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "frame-ancestors 'none';"
    )
    return response

STATIC_DIR = Path(__file__).parent / "static"

# Patterns de prompt injection courants
_INJECTION_RE = re.compile(
    r"(ignore\s+(previous|all|prior)\s+instructions?"
    r"|system\s*prompt"
    r"|<\s*/?system\s*>"
    r"|###\s*(system|instruction)"
    r"|tu\s+es\s+maintenant"
    r"|oublie\s+(tout|tes\s+instructions)"
    r"|forget\s+(your|all|previous)\s+instructions?)",
    re.IGNORECASE,
)

BLOCKED_UA = re.compile(r"(python-requests|curl|wget|scrapy|httpx|go-http|java/)", re.IGNORECASE)


def _check_bot(request: Request) -> None:
    ua = request.headers.get("user-agent", "")
    if not ua or BLOCKED_UA.search(ua):
        logger.warning("bot bloque ua=%r ip=%s", ua, request.client.host)
        raise HTTPException(status_code=403, detail="Forbidden")


def _sanitize_question(q: str) -> str:
    # Supprime les caractères de contrôle
    q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", q)
    if _INJECTION_RE.search(q):
        raise HTTPException(status_code=400, detail="Question non autorisée.")
    return q.strip()


ALLOWED_MODELS = {MISTRAL_MODEL, LM_STUDIO_MODEL}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    model: str | None = Field(default=None, max_length=80)
    n_results: int = Field(default=N_RESULTS, ge=1, le=10)
    filter_source: str | None = None


@app.get("/favicon.svg")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def list_models():
    model = MISTRAL_MODEL if LLM_MODE == "mistral" else LM_STUDIO_MODEL
    return {"models": [model], "mode": LLM_MODE}


@app.post("/api/ask/stream")
@limiter.limit("10/minute")
def ask_stream(request: Request, req: AskRequest):
    _check_bot(request)
    req.question = _sanitize_question(req.question)
    if req.model and req.model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Modèle non autorisé.")

    # filter_source doit être une clé connue
    if req.filter_source and req.filter_source not in SOURCE_MAP:
        req.filter_source = None

    logger.info("question=%r source=%s ip=%s", req.question[:80], req.filter_source, request.client.host)

    def generate():
        # 1. Retrieval
        fs = SOURCE_MAP.get(req.filter_source) if req.filter_source else None
        hits = search(req.question, n_results=req.n_results, filter_source=fs)
        context = build_context(hits)

        sources = [
            {
                "id": h["id"],
                "type": h["metadata"]["type"],
                "number": h["metadata"]["number"],
                "title": h["metadata"].get("title", ""),
                "source": h["metadata"].get("source", ""),
                "score": round(1 - h["distance"], 3),
            }
            for h in hits
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # 2. Generation
        active_model = req.model or (MISTRAL_MODEL if LLM_MODE == "mistral" else LM_STUDIO_MODEL)
        user_message = (
            f"Voici les extraits reglementaires pertinents :\n\n{context}\n\n---\n"
            f"Question : {req.question}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ]

        from openai import OpenAI
        if LLM_MODE == "mistral":
            key = os.environ.get("MISTRAL_API_KEY", "")
            client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=key, timeout=120.0)
        else:
            client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=120.0)

        stream = client.chat.completions.create(
            model=active_model, max_tokens=1024, stream=True, messages=messages
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""
            if content:
                yield f"data: {json.dumps({'type': 'token', 'token': content})}\n\n"
            elif reasoning:
                yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
