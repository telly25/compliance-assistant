"""
Pipeline RAG : retrieval semantique + generation via LM Studio (local).

LM Studio expose une API compatible OpenAI sur http://localhost:1234/v1.
Aucune cle API ni connexion internet requise pour la generation.

Pour changer de modele : modifier LM_STUDIO_MODEL ou passer model= a ask().
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from ingest.embed import search

# Modele charge dans LM Studio — adapter au modele que tu as charge
LM_STUDIO_MODEL = os.environ.get("LM_MODEL", "mistral")
LM_STUDIO_URL = os.environ.get("LM_URL", "http://localhost:1234/v1")
MAX_TOKENS = 1024
N_RESULTS = 3

SYSTEM_PROMPT = """\
Tu es un assistant expert en conformite reglementaire, specialise dans le RGPD \
(Reglement General sur la Protection des Donnees - UE 2016/679).

Tes reponses sont structurees, precises et operationnelles. Tu cites toujours \
les articles pertinents. Tu n'inventes pas d'obligations qui n'existent pas dans \
les textes fournis.

Format de reponse :
1. **Synthese** - resume en 2-3 phrases
2. **Obligations applicables** - liste des exigences concretes
3. **Checklist operationnelle** - actions a mener
4. **Articles de reference** - numeros et titres des articles cites
5. **Points d'attention** - risques ou zones grises eventuels
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    input_tokens: int
    output_tokens: int


def build_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        meta = h["metadata"]
        label = (
            f"Article {meta['number']} - {meta['title']}"
            if meta["type"] == "article"
            else f"Considerant {meta['number']}"
        )
        if meta.get("chapter"):
            label += f" (Chapitre {meta['chapter']} : {meta['chapter_title']})"
        parts.append(f"### {label}\n{h['text']}")
    return "\n\n".join(parts)


def ask(
    question: str,
    n_results: int = N_RESULTS,
    filter_type: str | None = None,
    verbose: bool = False,
    model: str = LM_STUDIO_MODEL,
) -> RAGResponse:
    """Pose une question au pipeline RAG.

    Args:
        question    : question en langage naturel
        n_results   : nombre d'extraits RGPD recuperes
        filter_type : restreindre a "article" ou "recital"
        verbose     : affiche les sources dans le terminal
        model       : nom du modele charge dans LM Studio

    Returns:
        RAGResponse avec la reponse et les tokens utilises
    """
    # 1. Retrieval semantique
    hits = search(question, n_results=n_results, filter_type=filter_type)
    context = build_context(hits)

    if verbose:
        print("\n-- Sources recuperees --")
        for h in hits:
            meta = h["metadata"]
            label = (
                f"Article {meta['number']}"
                if meta["type"] == "article"
                else f"Considerant {meta['number']}"
            )
            print(f"  [{h['id']}] {label} (score={1 - h['distance']:.3f})")
        print()

    # 2. Generation via LM Studio (streaming pour afficher mot par mot)
    client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    print("[...] Generation en cours...\n")

    stream = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Voici les extraits du RGPD pertinents pour ta reponse :\n\n"
                    f"{context}\n\n"
                    "---\n"
                    f"Question : {question}"
                ),
            },
        ],
    )

    answer_parts = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        answer_parts.append(delta)
    print()

    return RAGResponse(
        answer="".join(answer_parts),
        sources=hits,
        input_tokens=0,
        output_tokens=0,
    )


def interactive_session() -> None:
    """Lance une session de questions-reponses interactives."""
    print(f"Assistant conformite RGPD | modele : {LM_STUDIO_MODEL} | {LM_STUDIO_URL}")
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
        result = ask(question, verbose=True)
        print(result.answer)
        print(f"\n[tokens] prompt={result.input_tokens} | completion={result.output_tokens}\n")
