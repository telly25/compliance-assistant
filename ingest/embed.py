"""
Vectorise les chunks parsés et les stocke dans ChromaDB.

Une seule collection "regulations" pour tous les référentiels.
Le champ metadata "source" permet de filtrer par règlement.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ingest.sources import SOURCES

PARSED_DIR = Path(__file__).parent.parent / "data" / "parsed"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

COLLECTION_NAME = "regulations"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64


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

    # Verifie si deja indexe
    if not force:
        existing = collection.get(where={"source": source["name"]}, limit=1)
        if existing["ids"]:
            print(f"[embed] {source['name']} deja indexe ({collection.count()} docs total). Utilisez --force pour reindexer.")
            return collection

    print(f"[embed] Chargement du modele '{EMBED_MODEL}'...")
    model = SentenceTransformer(EMBED_MODEL)

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

    print(f"[embed] Vectorisation de {len(texts)} chunks {source['name']}...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

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
    """Recherche sémantique dans tous les référentiels (ou un seul).

    Args:
        query         : question en langage naturel
        n_results     : nombre de résultats
        filter_source : "RGPD" | "DORA" | None (tous)
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    collection = get_collection(chroma_dir)
    where = {"source": filter_source} if filter_source else None

    results = collection.query(
        query_embeddings=query_embedding,
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
