"""
RETRIEVER - Phase 2 Hybrid Retrieval + Re-ranking

Three-stage pipeline:

  BM25 (20)  +  Vector (20)   <- wide candidate nets
       |              |
       ----RRF fusion----
        (up to 40 unique chunks)
                |
       cross-encoder re-scores top 20
                |
           top-5 hits -> LLM

Key design decisions
- BM25 built in-memory from Chroma, cached for process lifetime (no extra storage).
- Vector search reuses the same Chroma + Gemini embeddings as Phase 1.
- RRF: score += 1/(rrf_k + rank) for each list, merge and sort.
- CrossEncoder reads (query, chunk) together; full cross-attention -> higher precision.
- Inverted index: pre-computed token-to-chunks mapping for fast candidate lookup.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    BM25_TOP_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    CROSS_ENCODER_MODEL,
    RERANK_POOL,
    RERANK_TOP_N,
    RRF_K,
    VECTOR_TOP_K,
)
from ingest import get_embeddings


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric characters.

    Intentionally simple: BM25 shines on exact tokens (error codes, model numbers,
    names) and a heavy NLP tokenizer adds latency with negligible gain here.
    """
    return re.sub(r"[^a-z0-9]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# INVERTED INDEX
# ---------------------------------------------------------------------------

class InvertedIndex:
    """Pre-computed token-to-chunks mapping for fast candidate lookup.

    Instead of scoring ALL chunks at query time, we only score chunks that
    contain at least one query token. This is O(m) where m = matching chunks,
    instead of O(n) where n = total chunks.

    Example
    -------
    >>> idx = InvertedIndex(["hello world", "hello python", "python rocks"])
    >>> idx.get_candidates(["hello"])
    {0, 1}  # only chunks 0 and 1 contain "hello"
    """

    def __init__(self, tokenized_docs: list[list[str]]):
        """Build inverted index from tokenized documents.

        Parameters
        ----------
        tokenized_docs : list of token lists, one per document
        """
        self._index: dict[str, set[int]] = defaultdict(set)
        self._num_docs = len(tokenized_docs)

        for doc_idx, tokens in enumerate(tokenized_docs):
            for token in set(tokens):  # dedupe within doc
                self._index[token].add(doc_idx)

    def get_candidates(self, query_tokens: list[str]) -> set[int]:
        """Return set of document indices containing ANY query token.

        Parameters
        ----------
        query_tokens : tokenized query

        Returns
        -------
        set of document indices that contain at least one query token
        """
        candidates = set()
        for token in query_tokens:
            candidates.update(self._index.get(token, set()))
        return candidates

    @property
    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return {
            "num_docs": self._num_docs,
            "num_tokens": len(self._index),
            "avg_postings": (
                sum(len(v) for v in self._index.values()) / len(self._index)
                if self._index else 0
            ),
        }


@lru_cache(maxsize=1)
def _get_bm25_index() -> tuple[BM25Okapi, list[dict[str, Any]], InvertedIndex]:
    """Build (or return cached) BM25 index + inverted index over ALL chunks in Chroma.

    Returns
    -------
    bm25    : BM25Okapi      - the index, ready to score queries
    chunks  : list[dict]     - parallel list of {text, metadata} dicts
    inv_idx : InvertedIndex  - pre-computed token-to-chunks mapping
    """
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    result    = vs.get(include=["documents", "metadatas"])
    texts     = result["documents"]
    metadatas = result["metadatas"]

    if not texts:
        raise RuntimeError("Vector store is empty - run: python main.py ingest")

    tokenized = [_simple_tokenize(t) for t in texts]
    bm25      = BM25Okapi(tokenized)
    inv_idx   = InvertedIndex(tokenized)
    chunks    = [{"text": t, "metadata": m} for t, m in zip(texts, metadatas)]

    stats = inv_idx.stats
    print(f"[BM25] Index built over {len(chunks)} chunks "
          f"| {stats['num_tokens']} unique tokens "
          f"| avg {stats['avg_postings']:.1f} docs/token")
    return bm25, chunks, inv_idx


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    """Load cross-encoder from HuggingFace (cached after first call).

    ms-marco-MiniLM-L-6-v2:
      - 22 M params, ~65 MB download
      - Trained on MS-MARCO passage ranking -> directly applicable to RAG
      - ~100-300 ms for 20 (query, chunk) pairs on CPU
    """
    print(f"[Reranker] Loading cross-encoder '{CROSS_ENCODER_MODEL}' ...")
    return CrossEncoder(CROSS_ENCODER_MODEL)


# ---------------------------------------------------------------------------
# SEARCH LEGS
# ---------------------------------------------------------------------------

def bm25_search(query: str, k: int = BM25_TOP_K) -> list[dict[str, Any]]:
    """Keyword search via BM25Okapi with inverted index optimization.

    Uses the inverted index to find candidate chunks containing at least one
    query token, then scores only those candidates with BM25. Falls back to
    scoring all chunks if no candidates found (empty query edge case).

    Returns list of {text, metadata, score, rank_source} where score is the
    raw BM25 relevance score (higher = better match).
    """
    bm25, chunks, inv_idx = _get_bm25_index()
    q_tokens = _simple_tokenize(query)

    # Use inverted index to find candidate chunks (only those with matching tokens)
    candidate_ids = inv_idx.get_candidates(q_tokens)

    if candidate_ids:
        # Score only candidates (O(m) instead of O(n))
        candidate_list = sorted(candidate_ids)
        scores = bm25.get_scores(q_tokens)
        scored = [(i, scores[i]) for i in candidate_list]
    else:
        # Fallback: no matching tokens found, score all chunks
        scores = bm25.get_scores(q_tokens)
        scored = [(i, scores[i]) for i in range(len(scores))]

    # Sort by score descending, take top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    return [
        {
            "text":        chunks[i]["text"],
            "metadata":    dict(chunks[i]["metadata"]),
            "score":       round(float(s), 6),
            "rank_source": "bm25",
        }
        for i, s in top
    ]


def vector_search(query: str, k: int = VECTOR_TOP_K) -> list[dict[str, Any]]:
    """Semantic search via Chroma + Gemini embeddings (same as Phase 1).

    Returns list of {text, metadata, score, rank_source} where score is cosine
    similarity (1 = identical).
    """
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    pairs = vs.similarity_search_with_score(query, k=k)
    return [
        {
            "text":        doc.page_content,
            "metadata":    dict(doc.metadata),
            "score":       round(1 - dist, 4),
            "rank_source": "vector",
        }
        for doc, dist in pairs
    ]


# ---------------------------------------------------------------------------
# RECIPROCAL RANK FUSION
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists via RRF.

    RRF score:  sum_i  1 / (rrf_k + rank_i)
    Deduplication key: first 200 chars of chunk text.

    Parameters
    ----------
    result_lists : list of ranked chunk dicts from each retrieval leg
    rrf_k        : smoothing constant (60 is the Elasticsearch default)
    """
    rrf_scores:   dict[str, float]          = {}
    chunk_by_key: dict[str, dict[str, Any]] = {}

    for ranked_list in result_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = chunk["text"][:200]
            rrf_scores[key]   = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            chunk_by_key.setdefault(key, chunk)

    return [
        {**chunk_by_key[k], "score": round(rrf_scores[k], 6), "rank_source": "rrf"}
        for k in sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
    ]


# ---------------------------------------------------------------------------
# CROSS-ENCODER RE-RANKING
# ---------------------------------------------------------------------------

def rerank(
    query:      str,
    candidates: list[dict[str, Any]],
    top_n:      int = RERANK_TOP_N,
    pool:       int = RERANK_POOL,
) -> list[dict[str, Any]]:
    """Re-score candidates using a cross-encoder; return top_n.

    The cross-encoder reads (query, chunk) TOGETHER through a transformer,
    enabling full cross-attention - far more precise than comparing independent
    bi-encoder vectors but O(n) inference cost, hence the small pool size.

    Parameters
    ----------
    query      : user question
    candidates : RRF-ranked list (we take first  items)
    top_n      : final number to keep (fed to LLM context)
    pool       : how many candidates to pass to the cross-encoder
    """
    ce = _get_cross_encoder()
    pool_items = candidates[:pool]
    pairs      = [(query, c["text"]) for c in pool_items]
    scores     = ce.predict(pairs)

    scored = sorted(zip(pool_items, scores), key=lambda x: x[1], reverse=True)
    return [
        {**chunk, "score": round(float(s), 4), "rank_source": "cross-encoder"}
        for chunk, s in scored[:top_n]
    ]


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str) -> list[dict[str, Any]]:
    """Full Phase 2 retrieval: BM25 + vector -> RRF -> cross-encoder.

    Returns top-N {text, metadata, score, rank_source} dicts ready for the LLM.
    """
    print(f"\n[Hybrid] Query: {query!r}")

    bm25_hits   = bm25_search(query)
    vector_hits = vector_search(query)
    print(f"[Hybrid] BM25={len(bm25_hits)} | Vector={len(vector_hits)}")

    fused = reciprocal_rank_fusion([bm25_hits, vector_hits])
    print(f"[Hybrid] RRF fused -> {len(fused)} unique chunks")

    final = rerank(query, fused)
    print(f"[Hybrid] Cross-encoder -> top {len(final)} chunks\n")

    return final
