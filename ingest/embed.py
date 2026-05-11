"""
Vectorise les chunks via l'API Mistral (mistral-embed) et stocke dans ChromaDB.

Avantage : aucune dépendance ML locale (torch, sentence-transformers).
Nécessite : MISTRAL_API_KEY dans l'environnement.
"""

import json
import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from ingest.sources import SOURCES

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

COLLECTION_NAME = "regulations"
EMBED_MODEL = "mistral-embed"
MISTRAL_URL = "https://api.mistral.ai/v1"
BATCH_SIZE = 32  # limite API Mistral


def _mistral_client() -> OpenAI:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise EnvironmentError(
            "Variable MISTRAL_API_KEY manquante.\n"
            "  -> Obtenez une cle sur https://console.mistral.ai\n"
            "  -> Puis : $env:MISTRAL_API_KEY = 'votre-cle'"
        )
    return OpenAI(base_url=MISTRAL_URL, api_key=key)


def get_collection(chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Vectorise une liste de textes via Mistral embed (par batchs)."""
    all_embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in resp.data])
        if start + BATCH_SIZE < len(texts):
            time.sleep(0.5)  # evite le rate limit
    return all_embeddings


def embed_source(source_key: str, force: bool = False) -> chromadb.Collection:
    """Vectorise et indexe un référentiel dans la collection commune."""
    if source_key not in SOURCES:
        raise ValueError(f"Source inconnue '{source_key}'. Disponibles : {list(SOURCES)}")

    source = SOURCES[source_key]
    parsed_path = PARSED_DIR / f"{source_key}.json"

    if not parsed_path.exists():
        raise FileNotFoundError(
            f"Fichier JSON manquant : {parsed_path}\n"
            f"Lancez d'abord : python main.py ingest --source {source_key} parse"
        )

    chunks: list[dict] = json.loads(parsed_path.read_text(encoding="utf-8"))
    collection = get_collection()

    if not force:
        existing = collection.get(where={"source": source["name"]}, limit=1)
        if existing["ids"]:
            print(f"[embed] {source['name']} deja indexe ({collection.count()} docs total). Utilisez --force pour reindexer.")
            return collection

    client = _mistral_client()
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "type": c["type"],
            "number": c["number"],
            "title": c["title"],
            "chapter": c["chapter"],
            "chapter_title": c["chapter_title"],
            "source": c["source"],
        }
        for c in chunks
    ]

    print(f"[embed] Vectorisation de {len(texts)} chunks {source['name']} via Mistral API...")
    embeddings = _embed_texts(texts, client)

    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[embed] {source['name']} indexe. Total collection : {collection.count()} vecteurs.")
    return collection


def search(
    query: str,
    n_results: int = 5,
    filter_source: str | None = None,
    chroma_dir: Path = CHROMA_DIR,
) -> list[dict]:
    """Recherche sémantique dans tous les référentiels (ou un seul)."""
    client = _mistral_client()
    query_embedding = _embed_texts([query], client)[0]

    collection = get_collection(chroma_dir)
    where = {"source": filter_source} if filter_source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]
