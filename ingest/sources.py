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
    "nis2": {
        "name": "NIS2",
        "full_name": "Directive sur la sécurité des réseaux et des systèmes d'information",
        "celex": "32022L2555",
    },
    "aiact": {
        "name": "AI Act",
        "full_name": "Règlement sur l'Intelligence Artificielle",
        "celex": "32024R1689",
    },
}


def celex_url(celex: str) -> str:
    return (
        f"https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/"
        f"?uri=CELEX:{celex}&from=FR"
    )
