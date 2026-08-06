"""
ingest.py — PDF parsing, chunking, embedding, and Qdrant upsert.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "rag_documents"
VECTOR_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
QDRANT_PATH = str(Path(__file__).parent.parent / "qdrant_local")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _iter_pdf_chunks(pdf_path: Path) -> Iterator[dict]:
    doc = fitz.open(str(pdf_path))
    doc_name = pdf_path.name
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if not text.strip():
            continue
        for chunk in _chunk_text(text):
            yield {"doc_name": doc_name, "page_number": page_num, "text": chunk}
    doc.close()


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def ingest_pdfs(pdf_dir: str | Path, force: bool = False, client: QdrantClient | None = None) -> None:
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    if client is None:
        client = QdrantClient(path=QDRANT_PATH)
    ensure_collection(client)

    if force:
        client.delete_collection(COLLECTION_NAME)
        ensure_collection(client)
        print("Re-ingesting all PDFs (forced).")

    model = SentenceTransformer(EMBEDDING_MODEL)

    total_chunks = 0
    for pdf_path in pdf_files:
        chunks = list(_iter_pdf_chunks(pdf_path))
        if not chunks:
            print(f"  [skip] {pdf_path.name} — no extractable text.")
            continue

        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "doc_name": chunks[i]["doc_name"],
                    "page_number": chunks[i]["page_number"],
                    "text": chunks[i]["text"],
                },
            )
            for i in range(len(chunks))
        ]

        # Upload in batches of 256 to avoid oversized payloads
        batch_size = 256
        for batch_start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[batch_start : batch_start + batch_size],
            )

        total_chunks += len(chunks)
        print(f"  [ok] {pdf_path.name} — {len(chunks)} chunks ingested.")

    print(f"\nIngestion complete. Total chunks stored: {total_chunks}")
