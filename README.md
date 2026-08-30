# PDF RAG Bot

Local-first RAG pipeline with **hybrid retrieval + cross-encoder re-ranking**, inline citations back to source document and page.

**Stack**: Gemini embeddings (`gemini-embedding-001`) + Chroma vector store + Groq (`openai/gpt-oss-120b`) + BM25 keyword search + cross-encoder reranker.

## Features

- **Phase 1** — Pure vector search (Gemini → Chroma → Groq)
- **Phase 2** — Hybrid retrieval: BM25 (with inverted index) + vector → RRF fusion → cross-encoder re-rank
- Page-precise citations `[1]` → `Source: file.pdf | Page 3`
- Cosine distance explicitly configured for Chroma (HNSW space)
- Rate-limit-safe ingestion (batched embedding with 429 backoff + retry)
- Evaluation harness comparing Phase 1 vs Phase 2 metrics

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

# 2. Ask a one-shot question:
.venv/bin/python main.py query "what chunk size should I use?"
.venv/bin/python main.py query "what chunk size should I use?" --hybrid   # Phase 2

# 3. Interactive mode:
.venv/bin/python main.py chat
.venv/bin/python main.py chat --hybrid                                    # Phase 2

# 4. Compare Phase 1 vs Phase 2:
.venv/bin/python main.py evaluate
.venv/bin/python main.py evaluate --qa qa.json                            # custom QA pairs
.venv/bin/python main.py evaluate --verbose                               # per-question rows
```

## Pipeline

```
                           ┌──────────────────┐
                           │     main.py      │  CLI: argparse
                           └──────┬───────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 v                v                 v
            ┌─────────┐      ┌─────────┐      ┌──────────┐
            │ ingest │      │ rag.py  │      │evaluate  │
            │  .py    │      │(answer) │      │  .py     │
            └────┬────┘      └────┬────┘      └────┬─────┘
                 │                │                 │
                 v          ┌─────┴─────┐           │
            ┌─────────┐     │           │           │
            │ Chroma  │     v           v           │
            │ DB on   │  Phase 1    Phase 2         │
            │ disk    │  (vector)   (hybrid)        │
            └─────────┘     │           │           │
                            │     ┌─────┴─────┐     │
                            │     v           v     │
                            │  BM25 + Vector      │
                            │       |             │
                            │   RRF Fusion        │
                            │       |             │
                            │  Cross-Encoder      │
                            │       |             │
                            └───┬───┘             │
                                v                 v
                         ┌────────────┐   ┌────────────┐
                         │ Groq LLM   │   │  Metrics   │
                         │ (answer)   │   │  (HR/P/MRR)│
                         └────────────┘   └────────────┘
```

| Stage | File | Detail |
|---|---|---|
| Ingestion | `ingest.py` | Per-page extraction (`pypdf`), metadata = `(source, page)` |
| Chunking | `ingest.py` | `RecursiveCharacterTextSplitter`, 800-token chunks, 100-token overlap |
| Embedding | `ingest.py` | Gemini `gemini-embedding-001`, 768-dim |
| Storage | `ingest.py` | Chroma at `./chroma_db`, **cosine space** |
| Phase 1 retrieval | `rag.py` | Top-5 cosine similarity |
| Phase 2 retrieval | `retriever.py` | BM25 + vector → RRF → cross-encoder top-5 |
| Inverted index | `retriever.py` | Token → chunk mapping; BM25 scores only matching chunks (`_score_bm25_candidates`, O(m) instead of O(n)) |
| Generation | `rag.py` | Groq `openai/gpt-oss-120b`, citation-forced prompt |
| Evaluation | `evaluate.py` | Hit-rate@5, Precision@5, MRR (Phase 1 vs 2) |

## Config

All knobs live in `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | 800 / 100 | Chunking sizes |
| `TOP_K` | 5 | Phase 1 chunks to LLM |
| `BM25_TOP_K` / `VECTOR_TOP_K` | 20 / 20 | Phase 2 candidate net |
| `RRF_K` | 60 | RRF smoothing constant |
| `RERANK_POOL` / `RERANK_TOP_N` | 20 / 5 | Cross-encoder input/output size |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (~65 MB, CPU) |

## Notes

- **Re-ingesting** always deletes the old collection first (`delete_collection()`) — a clean rebuild. If you ever doubt the distance metric, delete `chroma_db/` and re-ingest.
- **BM25 inverted index**: built once per process (`@lru_cache`), reused across queries. If you re-ingest mid-session, restart the process to refresh it.
- **Copyright/security**: `.env` (API keys) is git-ignored.