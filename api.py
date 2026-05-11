"""
API FastAPI — compliance-assistant

Endpoints :
  GET  /                  -> sert l'interface web (static/index.html)
  POST /api/ask           -> reponse complete JSON
  POST /api/ask/stream    -> reponse en streaming (Server-Sent Events)
  GET  /api/models        -> liste les modeles disponibles dans LM Studio
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

from query.rag import LM_STUDIO_MODEL, LM_STUDIO_URL, N_RESULTS, SYSTEM_PROMPT, build_context
from ingest.embed import search

app = FastAPI(title="Compliance Assistant RGPD")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ── Modèles de requête ────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    model: str = LM_STUDIO_MODEL
    n_results: int = N_RESULTS


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def list_models():
    """Retourne les modeles charges dans LM Studio."""
    try:
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        models = client.models.list()
        return {"models": [m.id for m in models.data]}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest):
    """Reponse en streaming (Server-Sent Events)."""

    def generate():
        # 1. Retrieval
        hits = search(req.question, n_results=req.n_results)
        context = build_context(hits)

        # Envoie les sources d'abord
        sources = [
            {
                "id": h["id"],
                "type": h["metadata"]["type"],
                "number": h["metadata"]["number"],
                "title": h["metadata"].get("title", ""),
                "score": round(1 - h["distance"], 3),
            }
            for h in hits
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # 2. Generation en streaming
        client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
        user_message = (
            f"Voici les extraits du RGPD pertinents :\n\n{context}\n\n---\n"
            f"Question : {req.question} /no_think"
        )

        stream = client.chat.completions.create(
            model=req.model,
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            reasoning = getattr(delta, "reasoning_content", None) or ""

            # Reponse finale (Mistral, LLaMA, etc.)
            if content:
                yield f"data: {json.dumps({'type': 'token', 'token': content})}\n\n"
            # Reflexion interne Qwen3 — envoyee separement pour affichage discret
            elif reasoning:
                yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
