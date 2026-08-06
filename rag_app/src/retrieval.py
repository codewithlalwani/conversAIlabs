"""
retrieval.py — Query embedding and Qdrant similarity search.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ingest import COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH

TOP_K = 5
MIN_SCORE = 0.3

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def retrieve(query: str, client: QdrantClient | None = None) -> list[dict]:
    """Return top-k chunks relevant to *query*, each above MIN_SCORE."""
    model = _get_model()
    query_vector = model.encode(query, convert_to_numpy=True).tolist()

    if client is None:
        client = QdrantClient(path=QDRANT_PATH)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        score_threshold=MIN_SCORE,
        with_payload=True,
    )

    return [
        {
            "score": hit.score,
            "doc_name": hit.payload["doc_name"],
            "page_number": hit.payload["page_number"],
            "text": hit.payload["text"],
        }
        for hit in response.points
    ]
