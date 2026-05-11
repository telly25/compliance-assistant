"""
Télécharge le texte du RGPD depuis EUR-Lex et le sauvegarde localement.
Source officielle : CELEX:32016R0679
"""

import time
import requests
from pathlib import Path

EURLEX_URL = (
    "https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/"
    "?uri=CELEX:32016R0679&from=FR"
)
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_FILE = RAW_DIR / "rgpd.html"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; compliance-assistant/0.1; "
        "+https://github.com/yourname/compliance-assistant)"
    )
}


def fetch_rgpd(force: bool = False) -> Path:
    """Télécharge le HTML du RGPD si absent (ou si force=True).

    Returns:
        Chemin vers le fichier HTML sauvegardé.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and not force:
        print(f"[fetch] Déjà présent : {RAW_FILE}")
        return RAW_FILE

    print(f"[fetch] Téléchargement depuis EUR-Lex…")
    resp = requests.get(EURLEX_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    RAW_FILE.write_bytes(resp.content)
    print(f"[fetch] Sauvegardé ({RAW_FILE.stat().st_size // 1024} Ko) → {RAW_FILE}")
    time.sleep(1)  # politesse envers EUR-Lex
    return RAW_FILE


if __name__ == "__main__":
    fetch_rgpd()
