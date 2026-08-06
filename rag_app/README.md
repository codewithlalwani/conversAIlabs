# RAG Application — Legal PDF Q&A

## Architecture

```
PDFs → pymupdf (parse) → chunks → sentence-transformers (embed) → Qdrant (store)
                                                                         ↓
User Query → embed query → Qdrant similarity search → top-k chunks
                                                              ↓
                                             OpenRouter LLM → Answer + Citations
```

## Libraries Used

| Library | Purpose |
|---------|---------|
| `pymupdf` | Extract text page-by-page from PDFs |
| `sentence-transformers` | Generate 384-dim embeddings locally |
| `qdrant-client` | Interface with local Qdrant vector DB |
| `openai` (SDK) | Call OpenRouter API (OpenAI-compatible) |
| `python-dotenv` | Load API key from `.env` |

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU-friendly)  
**LLM:** `mistralai/mistral-7b-instruct:free` via OpenRouter

## Assumptions

- PDFs are text-based (not scanned images).
- All PDFs are placed in `cases_pdf/` at the project root.
- Qdrant runs in local file-based mode (`qdrant_local/` directory, no Docker required).
- Chunk size: ~500 characters with ~50-character overlap.
- A similarity score threshold of 0.3 is used; queries below it return "not available".

## How to Run

### 1. Prerequisites

- Python 3.11+
- Docker

### 2. Install dependencies

```bash
cd rag_app
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure API key

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your_key>
```

### 5. Run

```bash
# Auto-ingests PDFs on first run, then opens query loop
python src/main.py

# Force re-ingestion
python src/main.py --ingest
```

The application will ingest all PDFs in `cases_pdf/`, then prompt for questions. Every answer includes document name, page number, and the retrieved text snippet. If information is not found, it responds clearly that it is unavailable.
