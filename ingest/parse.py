"""
Parse le HTML EUR-Lex du RGPD en chunks structurés.

Structure réelle EUR-Lex :
  - Articles   : <div id="art_N">
  - Titres     : <div id="art_N.tit_1">
  - Chapitres  : <div id="cpt_I">, <div id="cpt_II">... (parent des articles)
  - Considérants : <div id="rct_N">
  - Contenu principal : <div id="docHtml">

Produit une liste de dicts :
  - id, type, number, title, chapter, chapter_title, text, source
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

RAW_FILE = Path(__file__).parent.parent / "data" / "raw" / "rgpd.html"
PARSED_FILE = Path(__file__).parent.parent / "data" / "parsed" / "rgpd.json"


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _roman_to_int(s: str) -> int:
    """Convertit un chiffre romain en entier (pour trier les chapitres)."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for c in reversed(s.upper()):
        v = vals.get(c, 0)
        total += v if v >= prev else -v
        prev = v
    return total


def parse_html(html_path: Path = RAW_FILE) -> list[dict]:
    soup = BeautifulSoup(
        html_path.read_text(encoding="utf-8", errors="replace"),
        "lxml",
    )

    chunks: list[dict] = []

    # ── 1. Considérants ──────────────────────────────────────────────────────
    for div in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        number = re.sub(r"^rct_", "", div["id"])
        text = _clean(div.get_text(separator=" "))
        # Supprimer le préfixe "(N)" du texte
        text = re.sub(r"^\(\d+\)\s*", "", text)
        if text:
            chunks.append({
                "id": f"recital-{number}",
                "type": "recital",
                "number": number,
                "title": "",
                "chapter": "",
                "chapter_title": "",
                "text": text,
                "source": "RGPD",
            })

    # ── 2. Articles ──────────────────────────────────────────────────────────
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        number_raw = re.sub(r"^art_", "", div["id"])
        number = "1" if number_raw == "premier" else number_raw

        # Titre de l'article (div frère ou enfant avec id="art_N.tit_1")
        tit_div = soup.find("div", id=f"{div['id']}.tit_1")
        title = _clean(tit_div.get_text()) if tit_div else ""

        # Texte complet de l'article (sans répéter le titre)
        text = _clean(div.get_text(separator=" "))
        # Supprimer le "Article N\nTitre\n" en tête
        text = re.sub(r"^Article\s+\S+\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^" + re.escape(title) + r"\s*", "", text) if title else text
        text = _clean(text)

        # Chapitre parent
        chapter_id = ""
        chapter_title = ""
        for parent in div.parents:
            pid = parent.get("id", "")
            if re.match(r"^cpt_[IVXLCDM]+$", pid, re.IGNORECASE):
                chapter_id = re.sub(r"^cpt_", "", pid).upper()
                # Titre du chapitre : premier <p> ou <div> enfant direct du cpt
                first_text = parent.find(["p", "div", "span"], recursive=False)
                if first_text:
                    chapter_title = _clean(first_text.get_text())
                    chapter_title = re.sub(
                        r"^CHAPITRE\s+[IVXLCDM]+\s*[–—\-]?\s*", "",
                        chapter_title, flags=re.IGNORECASE
                    )
                break

        if text:
            chunks.append({
                "id": f"article-{number}",
                "type": "article",
                "number": number,
                "title": title,
                "chapter": chapter_id,
                "chapter_title": chapter_title,
                "text": text,
                "source": "RGPD",
            })

    # Tri final : considérants (par numéro) puis articles (par numéro)
    recitals = sorted(
        [c for c in chunks if c["type"] == "recital"],
        key=lambda c: int(c["number"]),
    )
    articles = sorted(
        [c for c in chunks if c["type"] == "article"],
        key=lambda c: int(c["number"]) if c["number"].isdigit() else 0,
    )
    return recitals + articles


def parse_and_save(html_path: Path = RAW_FILE, out_path: Path = PARSED_FILE) -> list[dict]:
    if not html_path.exists() or html_path.stat().st_size < 10_000:
        raise FileNotFoundError(
            f"Fichier HTML manquant ou trop petit : {html_path}\n"
            "Lancez d'abord : python main.py ingest fetch"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = parse_html(html_path)

    if not chunks:
        raise ValueError(
            "Aucun chunk extrait. Inspectez data/raw/rgpd.html pour verifier le contenu."
        )

    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    articles = sum(1 for c in chunks if c["type"] == "article")
    recitals = sum(1 for c in chunks if c["type"] == "recital")
    print(f"[parse] {len(chunks)} chunks extraits -- {articles} articles, {recitals} considerants")
    print(f"[parse] Sauvegarde -> {out_path}")
    return chunks


if __name__ == "__main__":
    parse_and_save()
