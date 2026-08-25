# PDF RAG Bot - Phase 1

Local-first RAG pipeline: **Gemini embeddings + Chroma vector store + Groq Llama 3.3 70B**, with inline citations back to source document and page.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env    # then paste both API keys into .env
```

Keys:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/apikey

## Usage

```bash
# 1. Drop PDFs (or .txt/.md) into corpus/, then index them:
.venv/bin/python main.py ingest

# 2. Ask questions:
.venv/bin/python main.py query "what chunk size should I use?"

# 3. Or interactive mode:
.venv/bin/python main.py chat
```

## Pipeline

| Stage | File | Detail |
|---|---|---|
| Ingestion | `ingest.py` | Per-page text extraction (`pypdf`), metadata = `(source, page)` |
| Chunking | `chunker.py` | Sentence-aware packing to 500-800 tokens, ~100 token overlap |
| Embedding | `api_clients.py` | Gemini `text-embedding-004`, batched |
| Storage | Chroma | Persistent at `./chroma_db`, cosine space |
| Retrieval | `rag.py` | Top-5 cosine similarity |
| Generation | `api_clients.py` | Groq `llama-3.3-70b-versatile`, citation-forced prompt |

## Config

All knobs live in `config.py`: chunk sizes/overlap, top-k, model names.
