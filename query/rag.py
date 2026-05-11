"""
Pipeline RAG : retrieval sémantique + génération via Claude API.

Utilise le prompt caching d'Anthropic sur le contexte récupéré pour
réduire la latence et les coûts lors de questions répétées sur les mêmes articles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anthropic

from ingest.embed import search

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048
N_RESULTS = 6  # articles/considérants récupérés par requête

SYSTEM_PROMPT = """\
Tu es un assistant expert en conformité réglementaire, spécialisé dans le RGPD \
(Règlement Général sur la Protection des Données — UE 2016/679).

Tes réponses sont structurées, précises et opérationnelles. Tu cites toujours \
les articles pertinents. Tu n'inventes pas d'obligations qui n'existent pas dans \
les textes fournis.

Format de réponse par défaut :
1. **Synthèse** — résumé en 2-3 phrases
2. **Obligations applicables** — liste des exigences concrètes
3. **Checklist opérationnelle** — actions à mener
4. **Articles de référence** — numéros et titres des articles cités
5. **Points d'attention** — risques ou zones grises éventuels
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


def build_context(hits: list[dict]) -> str:
    """Formate les chunks récupérés en contexte lisible pour le LLM."""
    parts = []
    for h in hits:
        meta = h["metadata"]
        label = (
            f"Article {meta['number']} — {meta['title']}"
            if meta["type"] == "article"
            else f"Considérant {meta['number']}"
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
) -> RAGResponse:
    """Pose une question au pipeline RAG.

    Args:
        question    : question en langage naturel
        n_results   : nombre d'extraits RGPD récupérés
        filter_type : restreindre à "article" ou "recital"
        verbose     : affiche les sources dans le terminal

    Returns:
        RAGResponse avec la réponse et les métadonnées d'usage
    """
    # 1. Retrieval
    hits = search(question, n_results=n_results, filter_type=filter_type)
    context = build_context(hits)

    if verbose:
        print("\n── Sources récupérées ──────────────────────────────")
        for h in hits:
            meta = h["metadata"]
            label = f"Article {meta['number']}" if meta["type"] == "article" else f"Considérant {meta['number']}"
            print(f"  [{h['id']}] {label} (distance={h['distance']:.3f})")
        print("────────────────────────────────────────────────────\n")

    # 2. Génération avec prompt caching sur le contexte
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    # Le contexte RGPD est mis en cache (min 1024 tokens pour activer le cache)
                    {
                        "type": "text",
                        "text": (
                            f"Voici les extraits du RGPD pertinents pour ta réponse :\n\n"
                            f"{context}\n\n"
                            "---\n"
                        ),
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"Question : {question}",
                    },
                ],
            }
        ],
    )

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return RAGResponse(
        answer=response.content[0].text,
        sources=hits,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=cache_read,
    )


def interactive_session() -> None:
    """Lance une session de questions-réponses interactives."""
    print("Assistant conformité RGPD — tapez 'quitter' pour arrêter.\n")
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

        print("\n[…] Recherche en cours…\n")
        result = ask(question, verbose=True)
        print(result.answer)
        print(
            f"\n[usage] in={result.input_tokens} | out={result.output_tokens} "
            f"| cache_read={result.cache_read_tokens}\n"
        )
