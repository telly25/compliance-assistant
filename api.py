"""
API FastAPI — RegWatch

Endpoints :
  GET  /                  -> interface web
  POST /api/ask/stream    -> reponse en streaming (SSE)
  GET  /api/models        -> modele actif
"""

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import OpenAI

from query.rag import (
    MISTRAL_MODEL, MISTRAL_URL, LM_STUDIO_MODEL, LM_STUDIO_URL,
    LLM_MODE, N_RESULTS, SYSTEM_PROMPT, SOURCE_MAP, build_context
)
from ingest.embed import search

app = FastAPI(title="RegWatch")

STATIC_DIR = Path(__file__).parent / "static"


def _get_client() -> tuple[OpenAI, str]:
    if LLM_MODE == "mistral":
        key = os.environ.get("MISTRAL_API_KEY", "")
        return OpenAI(base_url=MISTRAL_URL, api_key=key, timeout=120.0), MISTRAL_MODEL
    return OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=120.0), LM_STUDIO_MODEL


from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    model: str | None = None
    n_results: int = N_RESULTS
    filter_source: str | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def list_models():
    client, model = _get_client()
    return {"models": [model], "mode": LLM_MODE}


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):

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
        client, active_model = _get_client()
        if req.model:
            active_model = req.model

        user_message = (
            f"Voici les extraits reglementaires pertinents :\n\n{context}\n\n---\n"
            f"Question : {req.question}"
        )

        stream = client.chat.completions.create(
            model=active_model,
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
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
