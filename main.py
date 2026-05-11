"""
CLI principal — compliance-assistant

Commandes disponibles :
  ingest          Télécharge, parse et vectorise le RGPD (pipeline complet)
  ingest fetch    Télécharge uniquement le HTML depuis EUR-Lex
  ingest parse    Parse le HTML en chunks JSON
  ingest embed    Vectorise les chunks et peuple ChromaDB
  ask "<question>" Pose une question au pipeline RAG
  chat            Lance une session interactive

Usage :
  python main.py ingest
  python main.py ask "Quelles sont les bases légales du traitement ?"
  python main.py chat
"""

import argparse
import os
import sys
from pathlib import Path


def cmd_ingest(args: argparse.Namespace) -> None:
    step = getattr(args, "step", None)

    if step in (None, "fetch"):
        from ingest.fetch import fetch_rgpd
        fetch_rgpd(force=args.force)

    if step in (None, "parse"):
        from ingest.parse import parse_and_save
        parse_and_save()

    if step in (None, "embed"):
        from ingest.embed import embed_and_store
        embed_and_store(force=args.force)


def cmd_ask(args: argparse.Namespace) -> None:
    _check_api_key()
    from query.rag import ask
    result = ask(args.question, verbose=args.verbose)
    print(result.answer)
    if args.verbose:
        print(
            f"\n[usage] input={result.input_tokens} | output={result.output_tokens} "
            f"| cache_read={result.cache_read_tokens}"
        )


def cmd_chat(_args: argparse.Namespace) -> None:
    _check_api_key()
    from query.rag import interactive_session
    interactive_session()


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[erreur] La variable d'environnement ANTHROPIC_API_KEY est manquante.\n"
            "  → export ANTHROPIC_API_KEY=sk-ant-…",
            file=sys.stderr,
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compliance-assistant",
        description="Assistant de veille conformité RGPD (prototype).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Pipeline d'ingestion (fetch → parse → embed)")
    p_ingest.add_argument(
        "step", nargs="?", choices=["fetch", "parse", "embed"],
        help="Étape spécifique (défaut : toutes)"
    )
    p_ingest.add_argument("--force", action="store_true", help="Force le re-téléchargement/réindexation")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = sub.add_parser("ask", help="Pose une question au pipeline RAG")
    p_ask.add_argument("question", help="Question en langage naturel")
    p_ask.add_argument("-v", "--verbose", action="store_true", help="Affiche les sources et l'usage API")
    p_ask.set_defaults(func=cmd_ask)

    # chat
    p_chat = sub.add_parser("chat", help="Session interactive")
    p_chat.set_defaults(func=cmd_chat)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
