"""Retrieval ablation: change one thing at a time and re-measure hit@5 / MRR.

Each step builds an in-memory index from ``data/`` (nothing on disk is touched)
and scores every answerable question in the evaluation set. A chunk counts as
correct when it contains the question's evidence span, so results stay
comparable across chunking strategies.

    python -m evaluation.retrieval_experiments            # full ladder
    python -m evaluation.retrieval_experiments --only baseline structured
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass

from evaluation.common import (
    RESULTS_DIR,
    first_relevant_rank,
    load_eval_set,
    retrieval_metrics,
    write_json,
)
from rag.embeddings import _load_model
from rag.ingest import DEFAULT_DATA_DIR, chunk_docs, load_docs, make_vectorstore
from rag.retrieval import HybridRetriever, RetrievalConfig, _load_cross_encoder


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    dedupe: bool = True
    chunking: str = "structured"
    mode: str = "hybrid"
    reranker: str | None = None
    embedding: str = "all-MiniLM-L6-v2"


LADDER = [
    Experiment("baseline", "Original: 500-char chunks, vector only, MiniLM, duplicate PDFs indexed",
               dedupe=False, chunking="fixed", mode="vector"),
    Experiment("dedupe", "+ skip byte-identical files", chunking="fixed", mode="vector"),
    Experiment("structured", "+ heading-aware chunks, tables whole, drug/section/page metadata", mode="vector"),
    Experiment("bm25_only", "(reference) keyword search alone", mode="bm25"),
    Experiment("hybrid", "+ BM25 + vector fused with RRF", mode="hybrid"),
    Experiment("rerank_minilm", "+ re-rank top 20 with ms-marco-MiniLM-L-6 cross-encoder",
               reranker="ms-marco-MiniLM"),
    Experiment("rerank_bge", "+ re-rank top 20 with bge-reranker-base (instead of MiniLM)",
               reranker="bge-reranker-base"),
    # Embedding comparison, isolated (structured chunks, vector only)
    Experiment("emb_bge_small_vec", "vector only, bge-small-en-v1.5", mode="vector", embedding="bge-small"),
    Experiment("emb_bge_base_vec", "vector only, bge-base-en-v1.5", mode="vector", embedding="bge-base"),
    Experiment("emb_e5_small_vec", "vector only, e5-small-v2", mode="vector", embedding="e5-small"),
    Experiment("emb_e5_base_vec", "vector only, e5-base-v2", mode="vector", embedding="e5-base"),
    # Embedding comparison inside the full pipeline (hybrid + MiniLM re-ranker)
    Experiment("full_bge_small", "hybrid + rerank, bge-small", reranker="ms-marco-MiniLM", embedding="bge-small"),
    Experiment("full_bge_base", "hybrid + rerank, bge-base", reranker="ms-marco-MiniLM", embedding="bge-base"),
    Experiment("full_e5_base", "hybrid + rerank, e5-base", reranker="ms-marco-MiniLM", embedding="e5-base"),
    # Follow-ups once e5-small won the isolated comparison
    Experiment("hybrid_e5_small", "hybrid (BM25 + vector), e5-small, no re-ranker", embedding="e5-small"),
    Experiment("hybrid_e5_small_rerank_bge", "hybrid, e5-small, bge-reranker-base",
               reranker="bge-reranker-base", embedding="e5-small"),
]


def run_experiment(exp: Experiment, cases: list[dict], k: int = 5) -> dict:
    started = time.perf_counter()
    chunks = chunk_docs(load_docs(DEFAULT_DATA_DIR, dedupe=exp.dedupe), exp.chunking)
    store = make_vectorstore(chunks, exp.embedding)
    retriever = HybridRetriever(
        store, RetrievalConfig(top_k=k, mode=exp.mode, candidates=20, reranker=exp.reranker)
    )
    build_seconds = time.perf_counter() - started

    ranks, per_case, latencies = [], [], []
    for case in cases:
        t0 = time.perf_counter()
        result = retriever.retrieve(case["question"])
        latencies.append(time.perf_counter() - t0)
        texts = [c.document.page_content for c in result.chunks]
        rank = first_relevant_rank(texts, case["evidence"]) if case["answerable"] else None
        if case["answerable"]:
            ranks.append(rank)
        per_case.append({
            "id": case["id"],
            "answerable": case["answerable"],
            "rank": rank,
            "best_vector_score": round(result.best_vector_score, 4),
            "best_rerank_score": None if result.best_rerank_score is None else round(result.best_rerank_score, 4),
            "top_pages": [c.document.metadata.get("page") for c in result.chunks],
            "unique_top_chunks": len({t for t in texts}),
        })

    del retriever, store
    _load_model.cache_clear()  # free models between steps (large rerankers on small machines)
    _load_cross_encoder.cache_clear()
    gc.collect()

    metrics = retrieval_metrics(ranks, k)
    metrics.update({
        "chunks": len(chunks),
        "avg_unique_in_top5": statistics.mean(p["unique_top_chunks"] for p in per_case),
        "index_build_s": round(build_seconds, 1),
        "avg_query_ms": round(1000 * statistics.mean(latencies), 1),
    })
    return {"experiment": exp.__dict__, "metrics": metrics, "cases": per_case}


def to_markdown(results: list[dict], k: int = 5) -> str:
    lines = [
        f"| Step | Change | Chunks | hit@1 | hit@{k} | MRR@{k} | unique/top{k} | query ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m, e = r["metrics"], r["experiment"]
        lines.append(
            f"| `{e['name']}` | {e['description']} | {m['chunks']} | {m['hit@1']:.0%} | "
            f"{m[f'hit@{k}']:.0%} | {m[f'mrr@{k}']:.3f} | {m['avg_unique_in_top5']:.1f} | {m['avg_query_ms']:.0f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="experiment names to run")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cases = load_eval_set()
    ladder = [e for e in LADDER if not args.only or e.name in args.only]
    results = []
    for exp in ladder:
        print(f"\n=== {exp.name}: {exp.description}", flush=True)
        result = run_experiment(exp, cases, args.k)
        m = result["metrics"]
        print(f"hit@1={m['hit@1']:.1%} hit@{args.k}={m[f'hit@{args.k}']:.1%} "
              f"mrr={m[f'mrr@{args.k}']:.3f} chunks={m['chunks']}", flush=True)
        results.append(result)

    suffix = "" if not args.only else "_" + "_".join(args.only)
    write_json(RESULTS_DIR / f"retrieval_experiments{suffix}.json", results)
    table = to_markdown(results, args.k)
    (RESULTS_DIR / f"retrieval_experiments{suffix}.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
