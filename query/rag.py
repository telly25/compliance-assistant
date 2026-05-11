"""
Pipeline RAG : retrieval semantique + generation.

En local  : LM Studio (http://localhost:1234/v1)
En prod   : Mistral API (https://api.mistral.ai/v1)

La variable LLM_MODE determine le mode :
  - "local"  (defaut) : LM Studio
  - "mistral"         : Mistral API (necessite MISTRAL_API_KEY)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from ingest.embed import search

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_MODE = os.environ.get("LLM_MODE", "local")  # "local" ou "mistral"

# Local (LM Studio)
LM_STUDIO_MODEL = os.environ.get("LM_MODEL", "mistralai/ministral-3-3b")
LM_STUDIO_URL   = os.environ.get("LM_URL",   "http://localhost:1234/v1")

# Mistral API
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
MISTRAL_URL   = "https://api.mistral.ai/v1"

MAX_TOKENS = 1024
N_RESULTS  = 3

SYSTEM_PROMPT = """\
Tu es un assistant expert en conformite reglementaire europeenne, specialise dans \
le RGPD (UE 2016/679), le DORA (UE 2022/2554), le NIS2 (UE 2022/2555) \
et l'AI Act (UE 2024/1689).

Tes reponses sont structurees, precises et operationnelles. Tu cites toujours \
les articles pertinents avec leur source (RGPD, DORA, NIS2 ou AI Act). \
Tu n'inventes pas d'obligations qui n'existent pas dans les textes fournis.

Format de reponse :
1. **Synthese** - resume en 2-3 phrases
2. **Obligations applicables** - liste des exigences concretes
3. **Checklist operationnelle** - actions a mener
4. **Articles de reference** - numeros et titres des articles cites
5. **Points d'attention** - risques ou zones grises eventuels
"""

SOURCE_MAP = {"rgpd": "RGPD", "dora": "DORA", "nis2": "NIS2", "aiact": "AI Act"}


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    input_tokens: int
    output_tokens: int


def _get_client(model_override: str | None = None) -> tuple[OpenAI, str]:
    """Retourne (client, model) selon le mode configuré."""
    if LLM_MODE == "mistral":
        key = os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError("MISTRAL_API_KEY manquant.")
        return OpenAI(base_url=MISTRAL_URL, api_key=key, timeout=120.0), model_override or MISTRAL_MODEL
    else:
        return OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=120.0), model_override or LM_STUDIO_MODEL


def build_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        meta = h["metadata"]
        label = (
            f"[{meta['source']}] Article {meta['number']} - {meta['title']}"
            if meta["type"] == "article"
            else f"[{meta['source']}] Considerant {meta['number']}"
        )
        if meta.get("chapter"):
            label += f" (Chapitre {meta['chapter']})"
        parts.append(f"### {label}\n{h['text']}")
    return "\n\n".join(parts)


def ask(
    question: str,
    n_results: int = N_RESULTS,
    filter_source: str | None = None,
    verbose: bool = False,
    model: str | None = None,
    stream_callback=None,
) -> RAGResponse:
    """Pose une question au pipeline RAG.

    Args:
        question        : question en langage naturel
        n_results       : nombre d'extraits recuperes
        filter_source   : filtrer par referentiel (rgpd, dora, nis2, aiact)
        verbose         : affiche les sources dans le terminal
        model           : surcharge le modele par defaut
        stream_callback : fonction appelee pour chaque token (optionnel)
    """
    # 1. Retrieval
    fs = SOURCE_MAP.get(filter_source) if filter_source else None
    hits = search(question, n_results=n_results, filter_source=fs)
    context = build_context(hits)

    if verbose:
        print("\n-- Sources recuperees --")
        for h in hits:
            meta = h["metadata"]
            label = f"Article {meta['number']}" if meta["type"] == "article" else f"Considerant {meta['number']}"
            print(f"  [{meta['source']}] {label} (score={1 - h['distance']:.3f})")
        print()

    # 2. Generation
    client, active_model = _get_client(model)
    user_message = (
        f"Voici les extraits reglementaires pertinents :\n\n"
        f"{context}\n\n---\n"
        f"Question : {question}"
    )

    print("[...] Generation en cours...\n")
    stream = client.chat.completions.create(
        model=active_model,
        max_tokens=MAX_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    answer_parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta
        content = delta.content or ""
        reasoning = getattr(delta, "reasoning_content", None) or ""
        token = content or reasoning

        if token:
            if stream_callback:
                stream_callback(token, is_thinking=bool(reasoning and not content))
            else:
                print(token, end="", flush=True)
            if content:
                answer_parts.append(content)

    print()
    return RAGResponse(
        answer="".join(answer_parts),
        sources=hits,
        input_tokens=0,
        output_tokens=0,
    )


def interactive_session(model: str | None = None) -> None:
    mode_label = f"Mistral API ({MISTRAL_MODEL})" if LLM_MODE == "mistral" else f"LM Studio ({LM_STUDIO_MODEL})"
    print(f"RegWatch | {mode_label}")
    print("Tapez 'quitter' pour arreter.\n")
    while True:
        try:
            question = input("Votre question : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break

        if question.lower() in {"quitter", "exit", "quit", "q"}:
            print("Au revoir.")
            break
        if not question:
            continue

        print("\n[...] Recherche en cours...\n")
        result = ask(question, verbose=True, model=model)
        print(result.answer)
        print()
