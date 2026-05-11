"""
Télécharge le texte du RGPD depuis EUR-Lex avec Playwright.

EUR-Lex est protégé par AWS WAF : un vrai navigateur (Chromium headless)
est nécessaire pour résoudre le challenge JS et récupérer le HTML.

Pré-requis (une seule fois) :
    playwright install chromium
"""

import asyncio
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_FILE = RAW_DIR / "rgpd.html"

EURLEX_URL = (
    "https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/"
    "?uri=CELEX:32016R0679&from=FR"
)

# Sélecteur CSS du conteneur principal du texte réglementaire sur EUR-Lex
CONTENT_SELECTOR = "#document1, .eli-main-title, .doc-ti"


async def _fetch_with_playwright(url: str, output: Path) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            locale="fr-FR",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        print(f"[fetch] Navigation vers EUR-Lex…")
        await page.goto(url, wait_until="networkidle", timeout=60_000)

        # Attendre que le contenu réglementaire soit présent
        try:
            await page.wait_for_selector(CONTENT_SELECTOR, timeout=15_000)
        except Exception:
            print("[fetch] Avertissement : sélecteur de contenu non trouvé, on sauvegarde quand même.")

        html = await page.content()
        output.write_text(html, encoding="utf-8")
        size_kb = output.stat().st_size // 1024
        print(f"[fetch] Sauvegarde ({size_kb} Ko) -> {output}")
        await browser.close()


def fetch_rgpd(force: bool = False) -> Path:
    """Télécharge le HTML du RGPD si absent (ou si force=True).

    Returns:
        Chemin vers le fichier HTML sauvegardé.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists() and RAW_FILE.stat().st_size > 10_000 and not force:
        print(f"[fetch] Déjà présent : {RAW_FILE}")
        return RAW_FILE

    try:
        asyncio.run(_fetch_with_playwright(EURLEX_URL, RAW_FILE))
    except Exception as e:
        print(
            f"\n[fetch] Erreur Playwright : {e}\n"
            "  -> Assurez-vous d'avoir lance : python -m playwright install chromium\n"
            "  -> Ou telechargez manuellement le HTML de :\n"
            f"    {EURLEX_URL}\n"
            f"  -> et sauvegardez-le dans : {RAW_FILE}\n"
        )
        raise

    return RAW_FILE


if __name__ == "__main__":
    fetch_rgpd()
