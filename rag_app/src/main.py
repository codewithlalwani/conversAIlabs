"""
main.py — CLI entry point for the RAG application.

Usage:
    python src/main.py                     # auto-ingest if needed, then query loop
    python src/main.py --ingest            # force re-ingestion, then query loop
    python src/main.py --pdf-dir /path     # override PDF directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from generate import generate_answer
from ingest import COLLECTION_NAME, QDRANT_PATH, ingest_pdfs
from retrieval import retrieve

load_dotenv()

DEFAULT_PDF_DIR = Path(__file__).parent.parent.parent / "cases_pdf"


def _collection_is_populated(client: QdrantClient) -> bool:
    try:
        info = client.get_collection(COLLECTION_NAME)
        return (info.points_count or 0) > 0
    except (UnexpectedResponse, Exception):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG application over legal PDFs.")
    parser.add_argument(
        "--ingest", action="store_true", help="Force re-ingestion of all PDFs."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Directory containing PDF files (default: ../cases_pdf relative to project root).",
    )
    args = parser.parse_args()

    client = QdrantClient(path=QDRANT_PATH)

    if args.ingest or not _collection_is_populated(client):
        print(f"Ingesting PDFs from: {args.pdf_dir}\n")
        ingest_pdfs(args.pdf_dir, force=args.ingest, client=client)
        print()

    print("RAG application ready. Type your question or 'exit' to quit.\n")

    while True:
        try:
            query = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        chunks = retrieve(query, client=client)
        answer = generate_answer(query, chunks)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
