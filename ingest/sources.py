"""
Registre des référentiels réglementaires supportés.

Pour ajouter un nouveau référentiel :
  1. Ajouter une entrée dans SOURCES avec son CELEX ID
  2. Lancer : python main.py ingest --source <cle>
"""

SOURCES: dict[str, dict] = {
    "rgpd": {
        "name": "RGPD",
        "full_name": "Règlement Général sur la Protection des Données",
        "celex": "32016R0679",
    },
    "dora": {
        "name": "DORA",
        "full_name": "Digital Operational Resilience Act",
        "celex": "32022R2554",
    },
}


def celex_url(celex: str) -> str:
    return (
        f"https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/"
        f"?uri=CELEX:{celex}&from=FR"
    )
