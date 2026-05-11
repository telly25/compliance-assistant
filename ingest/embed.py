"""
Vectorise les chunks parsés et les stocke dans ChromaDB (mode local persistant).

Modèle d'embedding : paraphrase-multilingual-mpnet-base-v2
  → 768 dimensions, supporte le français nativement, ~400 Mo.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

PARSED_FILE = Path(__file__).parent.parent / "data" / "parsed" / "rgpd.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma"

COLLECTION_NAME = "rgpd"
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 64


def get_collection(chroma_dir: Path = CHROMA_DIR) -> chromadb.Collection:
    """Retourne la collection ChromaDB (crée si absente)."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(
    parsed_path: Path = PARSED_FILE,
    chroma_dir: Path = CHROMA_DIR,
    force: bool = False,
) -> chromadb.Collection:
    """Charge les chunks, les vectorise et les insère dans ChromaDB.

    Si force=False et que la collection est déjà peuplée, skip.
    """
    collection = get_collection(chroma_dir)

    if collection.count() > 0 and not force:
        print(f"[embed] Collection '{COLLECTION_NAME}' déjà peuplée ({collection.count()} docs). "
              "Utilisez force=True pour réindexer.")
        return collection

    chunks: list[dict] = json.loads(parsed_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError(f"Aucun chunk trouvé dans {parsed_path}. Lancez d'abord parse.py.")

    print(f"[embed] Chargement du modèle '{EMBED_MODEL}'…")
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

    print(f"[embed] Vectorisation de {len(texts)} chunks par batchs de {BATCH_SIZE}…")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    # Upsert par batchs pour éviter les timeouts sur grandes collections
    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"[embed] {collection.count()} vecteurs stockés dans '{CHROMA_DIR}'")
    return collection


def search(
    query: str,
    n_results: int = 5,
    filter_type: str | None = None,
    chroma_dir: Path = CHROMA_DIR,
) -> list[dict]:
    """Recherche sémantique dans la collection.

    Args:
        query       : question ou mots-clés
        n_results   : nombre de résultats
        filter_type : "article" | "recital" | None (tous)

    Returns:
        Liste de dicts {id, text, metadata, distance}
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()

    collection = get_collection(chroma_dir)
    where = {"type": filter_type} if filter_type else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    embed_and_store()
