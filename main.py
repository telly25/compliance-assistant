import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
CLI principal — compliance-assistant

Commandes :
  ingest [--source rgpd|dora|all] [fetch|parse|embed] [--force]
  ask "<question>" [-v] [--model <nom>] [--source rgpd|dora]
  chat [--model <nom>]
  sources   Liste les référentiels disponibles

Exemples :
  python main.py ingest                          # RGPD complet
  python main.py ingest --source dora            # DORA complet
  python main.py ingest --source all             # Tous
  python main.py ask "Qu'est-ce qu'un TIERCE ?" --source dora
  python main.py chat --model mistralai/ministral-3-3b
"""

import argparse

from ingest.sources import SOURCES


def cmd_sources(_args) -> None:
    print("Referentiels disponibles :")
    for key, s in SOURCES.items():
        print(f"  {key:10} {s['name']:8} {s['full_name']}")


def cmd_ingest(args) -> None:
    keys = list(SOURCES.keys()) if args.source == "all" else [args.source]
    step = getattr(args, "step", None)

    for key in keys:
        print(f"\n== {SOURCES[key]['name']} ==")
        if step in (None, "fetch"):
            from ingest.fetch import fetch_source
            fetch_source(key, force=args.force)
        if step in (None, "parse"):
            from ingest.parse import parse_source
            parse_source(key)
        if step in (None, "embed"):
            from ingest.embed import embed_source
            embed_source(key, force=args.force)


def cmd_ask(args) -> None:
    from query.rag import ask
    result = ask(
        args.question,
        verbose=args.verbose,
        model=args.model,
        filter_source=args.source,
    )
    print(result.answer)
    if args.verbose:
        print(f"\n[tokens] prompt={result.input_tokens} | completion={result.output_tokens}")


def cmd_chat(args) -> None:
    from query.rag import interactive_session
    interactive_session(model=args.model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compliance-assistant",
        description="Assistant de veille conformite reglementaire (RGPD, DORA...).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # sources
    p_src = sub.add_parser("sources", help="Liste les referentiels disponibles")
    p_src.set_defaults(func=cmd_sources)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Pipeline d'ingestion")
    p_ingest.add_argument(
        "step", nargs="?", choices=["fetch", "parse", "embed"],
        help="Etape specifique (defaut : toutes)"
    )
    p_ingest.add_argument(
        "--source", default="rgpd",
        choices=list(SOURCES.keys()) + ["all"],
        help="Referentiel a ingerer (defaut : rgpd)"
    )
    p_ingest.add_argument("--force", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = sub.add_parser("ask", help="Pose une question")
    p_ask.add_argument("question")
    p_ask.add_argument("-v", "--verbose", action="store_true")
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument(
        "--source", default=None,
        choices=list(SOURCES.keys()),
        help="Filtrer par referentiel (defaut : tous)"
    )
    p_ask.set_defaults(func=cmd_ask)

    # chat
    p_chat = sub.add_parser("chat", help="Session interactive")
    p_chat.add_argument("--model", default=None)
    p_chat.set_defaults(func=cmd_chat)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
