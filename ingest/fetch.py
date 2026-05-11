"""
Télécharge un référentiel réglementaire depuis EUR-Lex avec Playwright.

Usage :
    fetch_source("rgpd")
    fetch_source("dora")
"""

import asyncio
from pathlib import Path

from ingest.sources import SOURCES, celex_url

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CONTENT_SELECTOR = "#document1, #docHtml, .eli-main-title"


async def _fetch(url: str, output: Path) -> None:
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
        print(f"[fetch] Navigation vers EUR-Lex...")
        await page.goto(url, wait_until="networkidle", timeout=60_000)
        try:
            await page.wait_for_selector(CONTENT_SELECTOR, timeout=15_000)
        except Exception:
            pass
        html = await page.content()
        output.write_text(html, encoding="utf-8")
        print(f"[fetch] Sauvegarde ({output.stat().st_size // 1024} Ko) -> {output}")
        await browser.close()


def fetch_source(source_key: str, force: bool = False) -> Path:
    """Télécharge le HTML d'un référentiel depuis EUR-Lex.

    Args:
        source_key : clé dans SOURCES (ex. "rgpd", "dora")
        force      : re-télécharge même si déjà présent
    """
    if source_key not in SOURCES:
        raise ValueError(f"Source inconnue '{source_key}'. Disponibles : {list(SOURCES)}")

    source = SOURCES[source_key]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = RAW_DIR / f"{source_key}.html"
    url = celex_url(source["celex"])

    if output.exists() and output.stat().st_size > 10_000 and not force:
        print(f"[fetch] Deja present : {output}")
        return output

    print(f"[fetch] {source['name']} ({source['celex']})...")
    try:
        asyncio.run(_fetch(url, output))
    except Exception as e:
        print(f"[fetch] Erreur : {e}")
        raise

    return output
