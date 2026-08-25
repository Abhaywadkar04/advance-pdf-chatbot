"""
CLI ENTRY POINT - ties everything together.

Usage:
  python main.py ingest                         index everything in corpus/
  python main.py query "question"               Phase 1: vector-only
  python main.py query "question" --hybrid      Phase 2: BM25+vector+rerank
  python main.py chat                           interactive (Phase 1)
  python main.py chat --hybrid                  interactive (Phase 2)
  python main.py evaluate                       compare hit-rate before/after
  python main.py evaluate --qa qa.json          use your own QA pairs
  python main.py evaluate --verbose             per-question detail rows

This file contains NO RAG logic; it only parses arguments, calls
the appropriate module, and pretty-prints the results.
"""

import argparse
import sys


def print_answer(query: str, k: int, hybrid: bool = False) -> None:
    """Ask the RAG chain and display the answer plus its sources."""
    from rag import answer

    resp, hits = answer(query, k, hybrid=hybrid)

    print("\n=== ANSWER ===")
    print(resp)

    print("\n=== SOURCES ===")
    for i, h in enumerate(hits, start=1):
        m       = h["metadata"]
        stage   = h.get("rank_source", "vector")
        score   = h["score"]
        print(f"[{i}] {m['source']} | page {m['page']} "
              f"| score {score:.4f} | via {stage}")


def chat(k: int, hybrid: bool = False) -> None:
    """Interactive mode: keep asking questions until you type exit."""
    mode = "hybrid (Phase 2)" if hybrid else "vector-only (Phase 1)"
    print(f"Interactive RAG chat [{mode}]. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        try:
            print_answer(q, k, hybrid=hybrid)
        except Exception as exc:
            print(f"[error] {exc}")
    print("Bye.")


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF RAG bot - Phase 1 & 2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="index all documents in corpus/")

    qp = sub.add_parser("query", help="ask a one-off question")
    qp.add_argument("question", nargs="+")

    cp = sub.add_parser("chat", help="interactive Q&A loop")

    # -k and --hybrid are shared by query and chat
    for p in (qp, cp):
        p.add_argument("-k", type=int, default=None,
                       help="top-k chunks (default from config, Phase 1 only)")
        p.add_argument("--hybrid", action="store_true",
                       help="use Phase 2: BM25 + vector + cross-encoder reranking")

    # evaluate subcommand
    ep = sub.add_parser("evaluate",
                        help="compare Phase 1 vs Phase 2 hit-rate/precision/MRR")
    ep.add_argument("--qa",      default=None,
                    help="path to JSON file with [{\"q\": ..., \"source\": ...}] pairs")
    ep.add_argument("--verbose", action="store_true",
                    help="print per-question detail rows")

    args = ap.parse_args()

    if args.cmd == "ingest":
        import ingest as ing
        ing.ingest()
    elif args.cmd == "query":
        from config import TOP_K
        print_answer(" ".join(args.question), args.k or TOP_K,
                     hybrid=args.hybrid)
    elif args.cmd == "chat":
        from config import TOP_K
        chat(args.k or TOP_K, hybrid=args.hybrid)
    else:   # evaluate
        from evaluate import run_evaluation
        run_evaluation(qa_file=args.qa, verbose=args.verbose)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
