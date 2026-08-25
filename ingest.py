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
"""

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
    # ------------------------------------------------------------------
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    try:
        vs.delete_collection()   # wipe old state -> re-ingest is always a clean rebuild
        vs = Chroma(collection_name=COLLECTION_NAME,
                    embedding_function=get_embeddings(),
                    persist_directory=str(CHROMA_DIR))
    except Exception:
        pass                      # collection didn't exist yet -> nothing to delete
    vs.add_documents(chunks)
    print(f"Done. {len(chunks)} chunks stored in '{COLLECTION_NAME}' at {CHROMA_DIR}.")


if __name__ == "__main__":
    ingest()
