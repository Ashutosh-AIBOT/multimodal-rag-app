"""
retriever.py — Retrieves and merges results from ChromaDB across modalities.
"""

from __future__ import annotations
from src.chroma_store import query_collection
from src.query_router import QueryType
from src.config import COLLECTION_TEXT, COLLECTION_IMAGES, COLLECTION_TABLES


def retrieve_all(
    query: str,
    query_types: list[QueryType],
    k: int = 3,
) -> list[dict]:
    """
    Query relevant ChromaDB collections and return merged results.

    Returns list of:
      {"content": str, "modality": str, "metadata": dict, "score": float}
    """
    results: list[dict] = []

    if QueryType.TEXT in query_types:
        try:
            hits = query_collection(COLLECTION_TEXT, query, n_results=k)
            for hit in hits:
                results.append({
                    "content": hit["document"],
                    "modality": "text",
                    "metadata": hit["metadata"],
                    "score": hit["distance"],
                    "source": f"text_chunk",
                })
        except Exception as e:
            print(f"[retriever] Text retrieval failed: {e}")

    if QueryType.IMAGE in query_types:
        try:
            hits = query_collection(COLLECTION_IMAGES, query, n_results=k)
            for hit in hits:
                results.append({
                    "content": hit["document"],
                    "modality": "image",
                    "metadata": hit["metadata"],
                    "score": hit["distance"],
                    "source": hit["metadata"].get("image_path", "image"),
                })
        except Exception as e:
            print(f"[retriever] Image retrieval failed: {e}")

    if QueryType.TABLE in query_types:
        try:
            hits = query_collection(COLLECTION_TABLES, query, n_results=k)
            for hit in hits:
                results.append({
                    "content": hit["document"],
                    "modality": "table",
                    "metadata": hit["metadata"],
                    "score": hit["distance"],
                    "source": hit["metadata"].get("table_id", "table"),
                })
        except Exception as e:
            print(f"[retriever] Table retrieval failed: {e}")

    return merge_and_rank(results)


def merge_and_rank(results: list[dict]) -> list[dict]:
    """De-duplicate and interleave results across modalities."""
    seen: set[str] = set()
    unique: list[dict] = []
    for r in results:
        key = r["content"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Bucket by modality
    buckets: dict[str, list] = {"text": [], "image": [], "table": []}
    for r in unique:
        m = r["modality"] if r["modality"] in buckets else "text"
        buckets[m].append(r)

    # Sort each bucket by score (lower cosine distance = better)
    for b in buckets.values():
        b.sort(key=lambda x: x["score"])

    # Interleave
    merged: list[dict] = []
    max_len = max((len(b) for b in buckets.values()), default=0)
    for i in range(max_len):
        for mod in ["text", "image", "table"]:
            if i < len(buckets[mod]):
                merged.append(buckets[mod][i])
    return merged
