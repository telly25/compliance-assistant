"""
Parse le HTML EUR-Lex d'un référentiel en chunks structurés.

Structure EUR-Lex (commune à tous les règlements) :
  - Articles    : <div id="art_N">
  - Titres      : <div id="art_N.tit_1">
  - Chapitres   : <div id="cpt_I">, <div id="cpt_II">...
  - Considérants: <div id="rct_N">
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ingest.sources import SOURCES

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_body(soup: BeautifulSoup):
    for selector in [{"id": "docHtml"}, {"id": "document1"}, {"id": "TexteOnly"}]:
        node = soup.find("div", selector)
        if node:
            return node
    body = soup.body
    if body is None:
        raise ValueError("Contenu introuvable dans le HTML. Re-telechargez le fichier.")
    return body


def parse_html(html_path: Path, source_name: str) -> list[dict]:
    soup = BeautifulSoup(
        html_path.read_text(encoding="utf-8", errors="replace"), "lxml"
    )

    chunks: list[dict] = []

    # ── Considérants ─────────────────────────────────────────────────────────
    for div in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        number = re.sub(r"^rct_", "", div["id"])
        text = _clean(div.get_text(separator=" "))
        text = re.sub(r"^\(\d+\)\s*", "", text)
        if text:
            chunks.append({
                "id": f"{source_name.lower()}-recital-{number}",
                "type": "recital",
                "number": number,
                "title": "",
                "chapter": "",
                "chapter_title": "",
                "text": text,
                "source": source_name,
            })

    # ── Articles ──────────────────────────────────────────────────────────────
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        number_raw = re.sub(r"^art_", "", div["id"])
        number = "1" if number_raw == "premier" else number_raw

        tit_div = soup.find("div", id=f"{div['id']}.tit_1")
        title = _clean(tit_div.get_text()) if tit_div else ""

        text = _clean(div.get_text(separator=" "))
        text = re.sub(r"^Article\s+\S+\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^" + re.escape(title) + r"\s*", "", text) if title else text
        text = _clean(text)

        chapter_id, chapter_title = "", ""
        for parent in div.parents:
            pid = parent.get("id", "")
            if re.match(r"^cpt_[IVXLCDM]+$", pid, re.IGNORECASE):
                chapter_id = re.sub(r"^cpt_", "", pid).upper()
                first = parent.find(["p", "div", "span"], recursive=False)
                if first:
                    chapter_title = re.sub(
                        r"^CHAPITRE\s+[IVXLCDM]+\s*[–—\-]?\s*", "",
                        _clean(first.get_text()), flags=re.IGNORECASE
                    )
                break

        if text:
            chunks.append({
                "id": f"{source_name.lower()}-article-{number}",
                "type": "article",
                "number": number,
                "title": title,
                "chapter": chapter_id,
                "chapter_title": chapter_title,
                "text": text,
                "source": source_name,
            })

    recitals = sorted(
        [c for c in chunks if c["type"] == "recital"],
        key=lambda c: int(c["number"]),
    )
    articles = sorted(
        [c for c in chunks if c["type"] == "article"],
        key=lambda c: int(c["number"]) if c["number"].isdigit() else 0,
    )
    return recitals + articles


def parse_source(source_key: str) -> list[dict]:
    """Parse un référentiel et sauvegarde en JSON."""
    if source_key not in SOURCES:
        raise ValueError(f"Source inconnue '{source_key}'. Disponibles : {list(SOURCES)}")

    source = SOURCES[source_key]
    html_path = RAW_DIR / f"{source_key}.html"
    out_path = PARSED_DIR / f"{source_key}.json"

    if not html_path.exists() or html_path.stat().st_size < 10_000:
        raise FileNotFoundError(
            f"Fichier HTML manquant : {html_path}\n"
            f"Lancez d'abord : python main.py ingest --source {source_key} fetch"
        )

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = parse_html(html_path, source["name"])

    if not chunks:
        raise ValueError(f"Aucun chunk extrait pour {source['name']}. Verifiez le HTML.")

    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    articles = sum(1 for c in chunks if c["type"] == "article")
    recitals = sum(1 for c in chunks if c["type"] == "recital")
    print(f"[parse] {source['name']} : {len(chunks)} chunks -- {articles} articles, {recitals} considerants")
    print(f"[parse] Sauvegarde -> {out_path}")
    return chunks
