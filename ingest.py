"""
INGESTION PIPELINE  (run once per corpus: `python main.py ingest`)

RAW FILES -> Document objects -> CHUNKS -> EMBEDDINGS -> CHROMA

1. PyPDFLoader / TextLoader   read files, one Document per PDF page.
2. RecursiveCharacterTextSplitter cuts pages into <=800-token chunks with
   ~100-token overlap, preferring paragraph/sentence boundaries.
3. GoogleGenerativeAIEmbeddings turns each chunk's text into a 768-number
   vector (this is the ONLY cloud call during ingestion).
4. Chroma stores chunk text + metadata + vector on disk for later search.

Every chunk keeps metadata: {"source": filename, "page": page number}
which is what makes CITATIONS possible at answer time.

RATE LIMITING
-------------
Gemini's free tier caps embedding requests per minute/day. Sending 70+
chunks to add_documents() in one shot can blow through that quota and
raise a 429 RESOURCE_EXHAUSTED error, aborting the whole ingest with
nothing saved. To avoid that, embeddings are now added in small batches
with a pause between batches, and any 429 triggers an exponential
backoff retry instead of crashing.
"""

import time

import tiktoken  # token counter (same tokenizer family the splitter uses)

# --- LangChain building blocks -------------------------------------------
from langchain_chroma import Chroma                          # vector store wrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # text->vector
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_DIR,
    COLLECTION_NAME,
    CORPUS_DIR,
    EMBED_DIM,
    EMBED_MODEL,
)

# ---------------------------------------------------------------------------
# RATE-LIMIT KNOBS
# ---------------------------------------------------------------------------
# How many chunks go into a single embed_documents() call. Smaller batches
# mean smaller/less bursty requests, at the cost of more round trips.
INGEST_BATCH_SIZE = 10

# Seconds to sleep between successful batches, to stay under requests/minute
# quotas even when nothing fails.
INGEST_BATCH_DELAY = 2.0

# Retry policy for a batch that hits a 429.
MAX_RETRIES = 5
INITIAL_BACKOFF = 5.0   # seconds; doubles each retry (5, 10, 20, 40, 80)


def load_documents():
    """
    STEP 1 - EXTRACT: read every supported file in corpus/.

    Returns a list of LangChain Documents. Each Document has:
      .page_content -> the extracted text
      .metadata     -> dict, PyPDFLoader gives us {"source": path, "page": n}
    One PDF page == one Document (keeps page-level citations exact).
    """
    docs = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for d in PyPDFLoader(str(path)).load():
                d.metadata["source"] = path.name                     # short name, not full path
                d.metadata["page"] = d.metadata.get("page", 0) + 1   # make page numbers 1-based
                docs.append(d)
        elif suffix in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
            for d in loaded:
                d.metadata["page"] = 1          # plain text has no pages -> fake "page 1"
                d.metadata["source"] = path.name
            docs.extend(loaded)
    return docs


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """The embedding model object. Shared by ingest (below) and rag.py."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        output_dimensionality=EMBED_DIM,   # ask Gemini for 768-dim vectors
    )


def create_chroma() -> Chroma:
    """Create a Chroma client with the project's shared settings.

    Uses cosine distance for vector search (explicitly configured).
    Chroma's default is L2 (Euclidean); score = 1 - dist then behaves
    like true cosine similarity in [-1, 1].
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """True if exc looks like a Gemini 429 RESOURCE_EXHAUSTED error.

    langchain_google_genai wraps the underlying google.genai ClientError in
    its own GoogleGenerativeAIError, so we check the string instead of
    importing google.genai's error classes directly (keeps this decoupled
    from that package's internals).
    """
    msg = str(exc)
    return "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _add_batch_with_retry(vs: Chroma, batch: list) -> None:
    """Add one batch of chunks to Chroma, retrying on 429s with backoff."""
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vs.add_documents(batch)
            return
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            print(f"  [rate limit] batch failed (attempt {attempt}/{MAX_RETRIES}), "
                  f"retrying in {backoff:.0f}s ...")
            time.sleep(backoff)
            backoff *= 2  # exponential backoff


def _add_documents_in_batches(
    vs: Chroma,
    chunks: list,
    batch_size: int = INGEST_BATCH_SIZE,
    delay: float = INGEST_BATCH_DELAY,
) -> None:
    """Embed+store chunks in small batches instead of one giant call.

    This keeps each request to Gemini small and paces requests with a sleep,
    so a large PDF (many chunks) doesn't fire a burst that trips the
    requests-per-minute quota. Each batch also retries on 429 with
    exponential backoff before giving up.
    """
    total = len(chunks)
    num_batches = (total + batch_size - 1) // batch_size

    for batch_num, start in enumerate(range(0, total, batch_size), start=1):
        batch = chunks[start:start + batch_size]
        print(f"  Embedding batch {batch_num}/{num_batches} "
              f"({len(batch)} chunks) ...")
        _add_batch_with_retry(vs, batch)

        # Pace ourselves between batches (skip the sleep after the last one).
        if batch_num < num_batches:
            time.sleep(delay)


def ingest() -> None:
    """Full rebuild: load -> split -> embed -> store."""
    raw = load_documents()
    if not raw:
        print(f"No documents found in {CORPUS_DIR}. Drop PDFs there first.")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # STEP 2 - CHUNK
    # `from_tiktoken_encoder` makes chunk_size count TOKENS, not chars.
    # "Recursive" means it tries separators in order:  "\n\n" (paragraphs)
    # -> "\n" (lines) -> ". " (sentences) -> words/characters as last resort,
    # which is how sentence boundaries are respected.
    # ------------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(raw)
    # each chunk still carries its parent page's metadata -> citations survive

    # quick size report so you can see the 500-800 target being hit
    enc = tiktoken.get_encoding("o200k_base")
    sizes = [len(enc.encode(c.page_content)) for c in chunks]
    print(f"Ingesting: {len(chunks)} chunks "
          f"| tokens min/avg/max = {min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")

    # ------------------------------------------------------------------
    # STEP 3+4 - EMBED & STORE
    # add_documents() internally: chunk.text --Gemini--> vector, then writes
    # (vector + text + metadata) into Chroma's on-disk index.
    # Done in small batches (see _add_documents_in_batches) to avoid
    # tripping Gemini's requests-per-minute quota on large PDFs.
    # ------------------------------------------------------------------
    vs = create_chroma()
    try:
        vs.delete_collection()   # wipe old state -> re-ingest is always a clean rebuild
    except Exception:
        pass                      # collection didn't exist yet -> nothing to delete
    vs = create_chroma()

    _add_documents_in_batches(vs, chunks)
    print(f"Done. {len(chunks)} chunks stored in '{COLLECTION_NAME}' at {CHROMA_DIR}.")


if __name__ == "__main__":
    ingest()