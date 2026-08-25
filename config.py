"""
CONFIG - every tunable knob of the RAG pipeline lives here.
Phase 2 additions: BM25 / RRF / cross-encoder knobs at the bottom.

Nothing "smart" happens in this file. It just:
1. loads your API keys from the .env file into environment variables
2. exposes paths, model names and sizes as plain constants
so every other module imports from one single source of truth.
"""

import os
from pathlib import Path

# Folder where this project lives (used to build all other paths).
PROJECT_ROOT = Path(__file__).parent


def _load_env() -> None:
    """
    Minimal .env loader.

    Reads lines like  KEY=value  from the .env file next to this script
    and injects them into process environment variables (os.environ).

    We use `setdefault` so real environment variables (e.g. set in your
    shell) always win over the .env file.
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        # skip blank lines, comments (#...) and lines without "="
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()

# ---------------------------------------------------------------------------
# CORPUS / STORAGE
# ---------------------------------------------------------------------------
CORPUS_DIR = PROJECT_ROOT / "corpus"       # you drop raw PDFs / txt files here
CHROMA_DIR = PROJECT_ROOT / "chroma_db"    # Chroma writes its vector DB here
COLLECTION_NAME = "rag_chunks"             # name of the table inside Chroma

# ---------------------------------------------------------------------------
# GROQ  -> generates the natural-language ANSWER
# ---------------------------------------------------------------------------
# LangChain's ChatOpenAI client can talk to ANY OpenAI-compatible API,
# we just point it at Groq's base URL instead of OpenAI's.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"         # current flagship chat model on Groq

# ---------------------------------------------------------------------------
# GOOGLE GEMINI -> converts TEXT into VECTOR EMBEDDINGS
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768                            # length of every vector (768 numbers)

# ---------------------------------------------------------------------------
# CHUNKING SIZES (in tokens, not characters!)
# ---------------------------------------------------------------------------
CHUNK_SIZE_TOKENS = 800     # upper bound per chunk
CHUNK_OVERLAP_TOKENS = 100  # repeated tail between consecutive chunks
# Why overlap? A fact whose sentence sits right at a chunk boundary would
# otherwise be cut in half; the overlapping window keeps it whole somewhere.

# ---------------------------------------------------------------------------
# RETRIEVAL  (Phase 1)
# ---------------------------------------------------------------------------
TOP_K = 5                  # how many chunks we feed to the LLM per question

# ---------------------------------------------------------------------------
# HYBRID RETRIEVAL + RE-RANKING  (Phase 2)
# ---------------------------------------------------------------------------
# Both BM25 and vector legs cast a wider net before fusion.
BM25_TOP_K   = 20   # candidates returned by the BM25 (keyword) leg
VECTOR_TOP_K = 20   # candidates returned by the vector (semantic) leg

# RRF constant — higher values dampen the influence of top-ranked items.
# 60 is the standard default used by Elasticsearch / academic literature.
RRF_K = 60

# After RRF, the cross-encoder re-scores these many fused candidates.
RERANK_POOL  = 20   # take up to this many from RRF before cross-encoder
RERANK_TOP_N = 5    # keep this many after cross-encoder (feeds the LLM)

# Local cross-encoder model (downloaded from HuggingFace on first use, ~65 MB).
# Runs on CPU; ~100-300 ms for 20 candidates — fast enough for a chatbot.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
