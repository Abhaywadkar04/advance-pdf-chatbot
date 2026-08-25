"""
EVALUATE — Phase 2: compare Phase 1 (vector-only) vs Phase 2 (hybrid + rerank)

Metrics computed for both pipelines:
  hit-rate@5   fraction of questions where ANY correct source is in top-5
  precision@5  fraction of top-5 chunks that are from the correct source
  MRR          Mean Reciprocal Rank (1/rank of first correct hit)

Usage
-----
  python main.py evaluate                  # auto-generate questions from corpus
  python main.py evaluate --qa qa.json     # supply your own [{"q":...,"source":...}]
  python main.py evaluate --verbose        # also print per-question detail

Output
------
  Prints a side-by-side table, e.g.

  Metric       Phase-1 (vector)  Phase-2 (hybrid+rerank)
  -----------  ----------------  -----------------------
  Hit-rate@5   0.70              0.90
  Precision@5  0.34              0.58
  MRR          0.55              0.78

  Improvement: +20 pp hit-rate, +24 pp precision, +0.23 MRR
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from config import CHROMA_DIR, COLLECTION_NAME, RERANK_TOP_N, TOP_K
from ingest import get_embeddings


# ---------------------------------------------------------------------------
# GROUND-TRUTH GENERATION
# ---------------------------------------------------------------------------

def _auto_generate_qa(n: int = 10) -> list[dict[str, str]]:
    """Sample n chunks from Chroma and fabricate simple questions.

    The "question" is constructed from the first sentence of the chunk so the
    answer is guaranteed to be in the chunk.  This is weak as an eval signal
    but gives a useful sanity-check without hand-labelling effort.

    Returns
    -------
    list of {"q": question, "source": expected_source_filename}
    """
    vs     = Chroma(collection_name=COLLECTION_NAME,
                   embedding_function=get_embeddings(),
                   persist_directory=str(CHROMA_DIR))
    result = vs.get(include=["documents", "metadatas"])
    texts, metas = result["documents"], result["metadatas"]

    if not texts:
        raise RuntimeError("Vector store empty - run: python main.py ingest")

    # Deduplicate by source so we cover every file
    seen_sources: set[str] = set()
    pool: list[tuple[str, str]] = []
    pairs = list(zip(texts, metas))
    random.shuffle(pairs)
    for text, meta in pairs:
        src = meta.get("source", "")
        if src and src not in seen_sources:
            seen_sources.add(src)
            first_sentence = text.strip().split(".")[0].strip()
            if len(first_sentence) > 20:
                pool.append((first_sentence, src))

    # If fewer unique sources than n, allow repeats
    if len(pool) < n:
        extras = random.choices(list(zip(texts, metas)), k=n - len(pool))
        for text, meta in extras:
            first_sentence = text.strip().split(".")[0].strip()
            pool.append((first_sentence, meta.get("source", "")))

    sample = pool[:n]
    return [{"q": q, "source": src} for q, src in sample]


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def _hits(hits: list[dict[str, Any]], expected_source: str) -> list[bool]:
    """For each hit, is it from the expected source?"""
    return [h["metadata"].get("source", "") == expected_source for h in hits]


def hit_rate(hits: list[dict[str, Any]], expected_source: str) -> float:
    """1 if any hit is from the correct source, else 0."""
    return float(any(_hits(hits, expected_source)))


def precision(hits: list[dict[str, Any]], expected_source: str) -> float:
    """Fraction of hits from the correct source."""
    if not hits:
        return 0.0
    return sum(_hits(hits, expected_source)) / len(hits)


def mrr(hits: list[dict[str, Any]], expected_source: str) -> float:
    """Reciprocal rank of first correct hit (0 if none)."""
    for rank, is_hit in enumerate(_hits(hits, expected_source), start=1):
        if is_hit:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# RETRIEVAL WRAPPERS  (deferred imports to keep startup fast)
# ---------------------------------------------------------------------------

def _vector_retrieve(query: str, k: int = TOP_K) -> list[dict[str, Any]]:
    """Phase 1: vector-only retrieval."""
    from rag import _retrieve
    return _retrieve({"question": query, "k": k})


def _hybrid_retrieve(query: str) -> list[dict[str, Any]]:
    """Phase 2: BM25 + vector + RRF + cross-encoder."""
    from retriever import hybrid_retrieve
    return hybrid_retrieve(query)


# ---------------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------------

def evaluate(
    qa_pairs: list[dict[str, str]] | None = None,
    verbose: bool = False,
) -> dict[str, dict[str, float]]:
    """Run both pipelines on qa_pairs; return metric dicts.

    Parameters
    ----------
    qa_pairs : list of {"q": question, "source": expected_source}.
               If None, auto-generates 10 questions from the corpus.
    verbose  : if True, print per-question detail rows

    Returns
    -------
    {"phase1": {"hit_rate": ..., "precision": ..., "mrr": ...},
     "phase2": {"hit_rate": ..., "precision": ..., "mrr": ...}}
    """
    if qa_pairs is None:
        print("[Eval] No QA file supplied - auto-generating 10 questions from corpus.")
        qa_pairs = _auto_generate_qa(10)

    n = len(qa_pairs)
    p1 = {"hit_rate": 0.0, "precision": 0.0, "mrr": 0.0}
    p2 = {"hit_rate": 0.0, "precision": 0.0, "mrr": 0.0}

    print(f"[Eval] Evaluating {n} questions ...\n")

    if verbose:
        header = f"{'#':>3}  {'Q':40}  {'Src':20}  {'P1-HR':6}  {'P2-HR':6}"
        print(header)
        print("-" * len(header))

    for i, pair in enumerate(qa_pairs, start=1):
        q   = pair["q"]
        src = pair["source"]

        h1 = _vector_retrieve(q)
        h2 = _hybrid_retrieve(q)

        hr1, pr1, mr1 = hit_rate(h1, src), precision(h1, src), mrr(h1, src)
        hr2, pr2, mr2 = hit_rate(h2, src), precision(h2, src), mrr(h2, src)

        p1["hit_rate"]  += hr1;  p1["precision"] += pr1;  p1["mrr"] += mr1
        p2["hit_rate"]  += hr2;  p2["precision"] += pr2;  p2["mrr"] += mr2

        if verbose:
            q_short = textwrap.shorten(q, 40)
            print(f"{i:>3}  {q_short:40}  {src[:20]:20}  {hr1:.0f}     {hr2:.0f}")

    # Average
    for m in p1:
        p1[m] /= n
        p2[m] /= n

    return {"phase1": p1, "phase2": p2}


def print_results(results: dict[str, dict[str, float]]) -> None:
    """Pretty-print the comparison table."""
    p1, p2 = results["phase1"], results["phase2"]

    header = f"{'Metric':<15}  {'Phase-1 (vector)':>18}  {'Phase-2 (hybrid+rerank)':>24}  {'Delta':>8}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for label, key in [("Hit-rate@5", "hit_rate"), ("Precision@5", "precision"), ("MRR", "mrr")]:
        v1    = p1[key]
        v2    = p2[key]
        delta = v2 - v1
        sign  = "+" if delta >= 0 else ""
        print(f"{label:<15}  {v1:>18.3f}  {v2:>24.3f}  {sign}{delta:>7.3f}")

    print(sep)
    print()
    # Summary sentence
    hr_pp  = (p2["hit_rate"]  - p1["hit_rate"])  * 100
    pr_pp  = (p2["precision"] - p1["precision"]) * 100
    mrr_d  =  p2["mrr"]       - p1["mrr"]
    print(f"Summary: {hr_pp:+.1f} pp hit-rate | {pr_pp:+.1f} pp precision | {mrr_d:+.3f} MRR")
    print()


def run_evaluation(qa_file: str | None = None, verbose: bool = False) -> None:
    """Entry point called from main.py."""
    qa_pairs = None
    if qa_file:
        path = Path(qa_file)
        if not path.exists():
            raise FileNotFoundError(f"QA file not found: {qa_file}")
        qa_pairs = json.loads(path.read_text())
        print(f"[Eval] Loaded {len(qa_pairs)} QA pairs from {qa_file}")

    results = evaluate(qa_pairs, verbose=verbose)
    print_results(results)
