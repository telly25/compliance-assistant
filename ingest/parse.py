"""
Parse le HTML EUR-Lex du RGPD en chunks structurés.

Produit une liste de dicts avec les champs :
  - id          : identifiant unique  (ex. "article-6", "recital-47")
  - type        : "recital" | "article" | "annex"
  - number      : numéro (str)
  - title       : titre de l'article (peut être vide pour les considérants)
  - chapter     : numéro du chapitre parent (str, peut être vide)
  - chapter_title: titre du chapitre parent
  - text        : texte brut du chunk
  - source      : "RGPD"
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "rgpd.html"
PARSED_FILE = Path(__file__).parent.parent / "data" / "parsed" / "rgpd.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_article_heading(tag: Tag) -> tuple[str, str] | None:
    """Retourne (numero, titre) si le tag est un titre d'article, sinon None."""
    text = _clean(tag.get_text())
    m = re.match(r"^Article\s+(\d+)\s*[–—-]?\s*(.*)", text, re.IGNORECASE)
    if m:
        return m.group(1), _clean(m.group(2))
    return None


def _is_chapter_heading(tag: Tag) -> tuple[str, str] | None:
    text = _clean(tag.get_text())
    m = re.match(r"^CHAPITRE\s+([IVXLCDM\d]+)\s*[–—-]?\s*(.*)", text, re.IGNORECASE)
    if m:
        return m.group(1), _clean(m.group(2))
    return None


def _is_recital_heading(tag: Tag) -> str | None:
    """Retourne le numéro si le tag est un numéro de considérant, sinon None."""
    text = _clean(tag.get_text())
    m = re.match(r"^\((\d+)\)$", text)
    if m:
        return m.group(1)
    return None


# ── parsers principaux ────────────────────────────────────────────────────────

def parse_html(html_path: Path = RAW_FILE) -> list[dict]:
    soup = BeautifulSoup(html_path.read_bytes(), "html.parser")

    chunks: list[dict] = []
    current_chapter = ""
    current_chapter_title = ""

    # EUR-Lex structure : le contenu est dans des divs/p avec classes spécifiques.
    # On itère sur tous les éléments block en ordre document.
    body = soup.find("div", {"id": "document1"}) or soup.body

    paragraphs = body.find_all(["p", "div", "h1", "h2", "h3", "h4"], recursive=True)

    i = 0
    # ── Phase 1 : considérants ──────────────────────────────────────────────
    recital_buffer: list[str] = []
    recital_number = ""
    in_recitals = False

    # ── Phase 2 : articles ──────────────────────────────────────────────────
    article_buffer: list[str] = []
    article_number = ""
    article_title = ""
    in_articles = False

    for tag in paragraphs:
        if tag.find_parent(["p", "div"]) and tag.name in ["p", "div"]:
            # évite la double-capture des éléments imbriqués
            parent = tag.parent
            if parent and parent.name in ["p", "div"] and parent in paragraphs:
                continue

        text = _clean(tag.get_text())
        if not text:
            continue

        # ── Détection chapitre ──
        chapter_match = _is_chapter_heading(tag)
        if chapter_match:
            # flush article en cours
            if article_buffer and article_number:
                chunks.append(_make_article(
                    article_number, article_title,
                    current_chapter, current_chapter_title,
                    article_buffer
                ))
                article_buffer, article_number, article_title = [], "", ""
            current_chapter, current_chapter_title = chapter_match
            in_articles = True
            continue

        # ── Détection article ──
        article_match = _is_article_heading(tag)
        if article_match:
            if article_buffer and article_number:
                chunks.append(_make_article(
                    article_number, article_title,
                    current_chapter, current_chapter_title,
                    article_buffer
                ))
            article_number, article_title = article_match
            article_buffer = []
            in_articles = True
            in_recitals = False
            continue

        # ── Détection début des considérants ──
        if re.match(r"ONT ARRÊTÉ LE PRÉSENT RÈGLEMENT|considérant ce qui suit", text, re.IGNORECASE):
            in_recitals = True
            in_articles = False
            continue

        # ── Fin des considérants (début de la partie normative) ──
        if re.match(r"^CHAPITRE I", text, re.IGNORECASE):
            if recital_buffer and recital_number:
                chunks.append(_make_recital(recital_number, recital_buffer))
                recital_buffer, recital_number = [], ""
            in_recitals = False
            in_articles = True
            current_chapter = "I"
            current_chapter_title = "DISPOSITIONS GÉNÉRALES"
            continue

        # ── Accumulation considérants ──
        if in_recitals:
            recital_num = _is_recital_heading(tag)
            if recital_num:
                if recital_buffer and recital_number:
                    chunks.append(_make_recital(recital_number, recital_buffer))
                recital_number = recital_num
                recital_buffer = []
            elif recital_number:
                recital_buffer.append(text)
            continue

        # ── Accumulation articles ──
        if in_articles and article_number:
            article_buffer.append(text)

    # flush final
    if recital_buffer and recital_number:
        chunks.append(_make_recital(recital_number, recital_buffer))
    if article_buffer and article_number:
        chunks.append(_make_article(
            article_number, article_title,
            current_chapter, current_chapter_title,
            article_buffer
        ))

    return chunks


def _make_article(number, title, chapter, chapter_title, lines) -> dict:
    return {
        "id": f"article-{number}",
        "type": "article",
        "number": number,
        "title": title,
        "chapter": chapter,
        "chapter_title": chapter_title,
        "text": " ".join(lines),
        "source": "RGPD",
    }


def _make_recital(number, lines) -> dict:
    return {
        "id": f"recital-{number}",
        "type": "recital",
        "number": number,
        "title": "",
        "chapter": "",
        "chapter_title": "",
        "text": " ".join(lines),
        "source": "RGPD",
    }


def parse_and_save(html_path: Path = RAW_FILE, out_path: Path = PARSED_FILE) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = parse_html(html_path)
    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    articles = sum(1 for c in chunks if c["type"] == "article")
    recitals = sum(1 for c in chunks if c["type"] == "recital")
    print(f"[parse] {len(chunks)} chunks extraits — {articles} articles, {recitals} considérants")
    print(f"[parse] Sauvegardé → {out_path}")
    return chunks


if __name__ == "__main__":
    parse_and_save()
