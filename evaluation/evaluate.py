"""End-to-end offline evaluation of the MediAsk RAG pipeline.

Measures, over evaluation/eval_set.jsonl:
  * Retrieval hit@5 / MRR@5   - is the evidence chunk in the top 5?
  * Answer correctness        - LLM-as-judge, 0-3 rubric vs the reference answer
  * Faithfulness              - LLM-as-judge, is every answer sentence supported by
                                the chunk it cites? (final answer and raw pre-guardrail answer)
  * Refusal accuracy          - does it say "not found" on unanswerable questions,
                                and how often does it wrongly refuse answerable ones?

It also writes a 10-question spot-check sheet for checking the judge by hand.

Built for a CPU-only laptop: the LLM runs through an API, only a small
embedding model runs locally, and every question is checkpointed to
evaluation/results/e2e_<label>.answers.jsonl / .judged.jsonl. Re-running the same
command resumes where it stopped (crash, Ctrl-C, or exhausted API quota).

    python -m evaluation.evaluate --label final --provider gemini --model gemini-3.5-flash-lite
    python -m evaluation.evaluate --label final ... --phase generate --ids q01 q02   # small batch
    python -m evaluation.evaluate --label final ... --phase judge                    # score saved answers
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

from evaluation.common import (
    DEFAULT_DATASET,
    NAIVE_PROMPT,
    RESULTS_DIR,
    naive_refused,
    first_relevant_rank,
    load_eval_set,
    retrieval_metrics,
    write_json,
)
from rag.guardrails import GuardrailSettings, _text_of, split_sentences, strip_citations
from rag.ingest import DEFAULT_DATA_DIR, chunk_docs, load_docs, make_vectorstore
from rag.rag_chain import MediAskPipeline, get_llm
from rag.retrieval import HybridRetriever, RetrievalConfig

CORRECTNESS_RUBRIC = """Part A - correctness. Compare the system answer with the reference answer.
Question: {question}
Reference answer: {reference}
Scoring rubric:
3 = Correct and complete: states every key fact of the reference (numbers, units, populations,
    conditions) and nothing that contradicts it.
2 = Mostly correct: the main fact is right but a secondary detail from the reference is missing.
1 = Partially correct: some relevant correct information, but the main fact is missing, vague,
    or one significant statement is wrong.
0 = Wrong: contradicts the reference, gives an incorrect number, or says the information is not available.
Judge meaning, not wording. Do not penalise extra information unless it is wrong.
"""

NO_CORRECTNESS = 'Part A - skip: set "score" and "reason" to null.\n'

CITED_RULE = """A sentence is SUPPORTED only if every factual claim in it is stated in, or directly
follows from, the documents it cites (bracketed ids at the end of the sentence). Claims from general
knowledge that the cited documents do not state are NOT supported. A sentence with no citation is NOT supported."""

UNCITED_RULE = """A sentence is SUPPORTED only if every factual claim in it is stated in, or directly
follows from, at least one of the documents. Claims from general knowledge that the documents do not
state are NOT supported. Sentences with no factual claim (e.g. "I don't know") count as supported."""

JUDGE_PROMPT = """You are evaluating one answer from a medical question-answering system.

{correctness}
Part B - faithfulness. For each numbered answer sentence decide whether it is supported.
{rule}

Documents:
{documents}

System answer, split into numbered sentences:
{sentences}

Return only JSON:
{{"score": <0|1|2|3 or null>, "reason": "<one sentence or null>",
  "verdicts": [{{"n": <sentence number>, "supported": true|false, "reason": "<short>"}}]}}"""


class NaivePipeline:
    """The original app: LangChain default prompt over the top-3 chunks, no guardrails."""

    def __init__(self, retriever: HybridRetriever, llm, context_k: int = 3):
        self.retriever, self.llm, self.context_k = retriever, llm, context_k

    def answer(self, question: str) -> dict:
        retrieval = self.retriever.retrieve(question)
        docs = retrieval.documents[: self.context_k]
        prompt = NAIVE_PROMPT.format(context="\n\n".join(d.page_content for d in docs), question=question)
        raw = _text_of(self.llm.invoke([HumanMessage(content=prompt)])).strip()
        refused = naive_refused(raw)
        return {
            "answer_text": raw,
            "raw_answer": raw,
            "refused": refused,
            "refusal_reason": "model_said_unknown" if refused else None,
            "unsupported_sentences": [],
            "retrieval": {"best_vector_score": retrieval.best_vector_score,
                          "best_rerank_score": retrieval.best_rerank_score},
            # all top-5 for hit@5; only the first 3 were shown to the model
            "sources": [{"id": n, "text": c.document.page_content, "page": c.document.metadata.get("page")}
                        for n, c in enumerate(retrieval.chunks, start=1)],
            "context_k": self.context_k,
        }


def parse_json(text: str) -> dict:
    """Return the first JSON object in ``text`` (judges sometimes add prose or code fences)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"No JSON object in judge output: {text[:200]!r}")


def call_with_retry(fn, *args, attempts: int = 6, base_delay: float = 5.0):
    for attempt in range(attempts):
        try:
            return fn(*args)
        except Exception as exc:  # provider SDKs raise many types; retry transient ones
            message = str(exc).lower()
            transient = any(
                word in message
                for word in ("429", "rate", "quota", "exhausted", "503", "500", "timeout",
                             "timed out", "unavailable", "overloaded", "deadline")
            )
            if not transient or attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"    transient error ({exc.__class__.__name__}); retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)


def judge_answer(judge, case: dict, answer: str, sources: list[dict], cited: bool, grade: bool) -> dict:
    """One judge call: correctness (if ``grade``) plus per-sentence faithfulness.

    ``cited=False`` is for answers without citations (the naive baseline): a
    sentence then counts as supported if *any* shown chunk supports it.
    """
    sentences = [s for s in split_sentences(answer) if strip_citations(s)]
    if not sentences and not grade:
        return {"correctness": None, "faithfulness": {"score": None, "verdicts": []}}
    prompt = JUDGE_PROMPT.format(
        correctness=CORRECTNESS_RUBRIC.format(question=case["question"], reference=case["reference_answer"])
        if grade else NO_CORRECTNESS,
        rule=CITED_RULE if cited else UNCITED_RULE,
        documents="\n\n".join(f"[{s['id']}] {s['text']}" for s in sources),
        sentences="\n".join(f"{n}. {s}" for n, s in enumerate(sentences, start=1)) or "(empty answer)",
    )
    result = parse_json(_text_of(call_with_retry(judge.invoke, prompt)))
    correctness = None
    if grade:
        correctness = {"score": int(result["score"]), "reason": result.get("reason") or ""}
    verdicts = result.get("verdicts") or []
    supported = sum(1 for v in verdicts if v.get("supported") is True)
    faithfulness = {
        "score": supported / len(sentences) if sentences else None,
        "verdicts": verdicts,
        "sentences": sentences,
    }
    return {"correctness": correctness, "faithfulness": faithfulness}


def build_pipeline(args, llm):
    chunks = chunk_docs(load_docs(DEFAULT_DATA_DIR, dedupe=not args.no_dedupe), args.chunking)
    store = make_vectorstore(chunks, args.embedding)
    retriever = HybridRetriever(
        store, RetrievalConfig(top_k=5, mode=args.mode, candidates=20, reranker=args.reranker)
    )
    if args.naive:
        return NaivePipeline(retriever, llm)
    settings = GuardrailSettings(
        min_vector_score=args.min_vector_score,
        min_rerank_score=args.min_rerank_score,
        citation_check=args.citation_check,
        citation_action=args.citation_action,
    )
    return MediAskPipeline(retriever, llm, settings)


# --------------------------------------------------------------------------- #
# Checkpoint files: one JSON object per line, appended as each question finishes
# --------------------------------------------------------------------------- #


def read_checkpoint(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written last line after a hard crash
                rows[row["id"]] = row
    return rows


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # After a hard crash the last line may be half-written with no newline;
    # start a new line so this row is not glued onto the broken one.
    needs_newline = path.exists() and path.stat().st_size > 0 and not path.read_bytes().endswith(b"\n")
    with path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            handle.write("\n")
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_row(pipeline, case: dict) -> dict:
    started = time.perf_counter()
    result = call_with_retry(pipeline.answer, case["question"])
    latency = time.perf_counter() - started
    texts = [s["text"] for s in result["sources"]]
    shown_k = result.get("context_k", len(result["sources"]))
    return {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "reference_answer": case["reference_answer"],
        "rank": first_relevant_rank(texts, case["evidence"]) if case["answerable"] else None,
        "answer_text": result["answer_text"],
        "raw_answer": result["raw_answer"],
        "refused": result["refused"],
        "refusal_reason": result["refusal_reason"],
        "unsupported_sentences": result["unsupported_sentences"],
        "retrieval": result["retrieval"],
        "source_pages": [s["page"] for s in result["sources"]],
        "shown_sources": [{"id": s["id"], "text": s["text"]} for s in result["sources"][:shown_k]],
        "latency_s": round(latency, 2),
    }


def judge_row(judge, row: dict, case: dict, cited: bool) -> dict:
    row = dict(row)
    shown = row["shown_sources"]
    if row["refused"]:
        if row["answerable"]:
            row["correctness"] = {"score": 0, "reason": "refused"}
    else:
        judged = judge_answer(judge, case, row["answer_text"], shown, cited, grade=row["answerable"])
        if row["answerable"]:
            row["correctness"] = judged["correctness"]
        row["faithfulness"] = judged["faithfulness"]

    raw = (row["raw_answer"] or "").strip()
    if raw and not raw.startswith("NOT_FOUND"):
        if row["refused"] or raw != row["answer_text"].strip():
            # The citation check changed the answer: also score what the model originally wrote.
            row["raw_faithfulness"] = judge_answer(judge, case, raw, shown, cited, grade=False)["faithfulness"]
        elif "faithfulness" in row:
            row["raw_faithfulness"] = row["faithfulness"]
    return row


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    ranks = [r["rank"] for r in answerable]
    correctness = [r["correctness"]["score"] for r in answerable if r.get("correctness")]
    answered = [r for r in rows if not r["refused"] and (r.get("faithfulness") or {}).get("score") is not None]
    raw_faith = [r["raw_faithfulness"]["score"] for r in rows
                 if (r.get("raw_faithfulness") or {}).get("score") is not None]
    refused_unanswerable = sum(1 for r in unanswerable if r["refused"])
    refused_answerable = sum(1 for r in answerable if r["refused"])
    correct_decisions = refused_unanswerable + (len(answerable) - refused_answerable)

    def share(items, predicate):
        return sum(1 for item in items if predicate(item)) / len(items) if items else None

    return {
        "cases": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        **retrieval_metrics(ranks, 5),
        "correctness_mean_0_3": statistics.mean(correctness) if correctness else None,
        "correctness_pct_score_ge_2": share(correctness, lambda s: s >= 2),
        "correctness_pct_score_3": share(correctness, lambda s: s == 3),
        "faithfulness_final": statistics.mean(r["faithfulness"]["score"] for r in answered) if answered else None,
        "faithfulness_raw_before_citation_check": statistics.mean(raw_faith) if raw_faith else None,
        "answers_fully_faithful_pct": share(answered, lambda r: r["faithfulness"]["score"] == 1.0),
        "sentences_dropped_by_citation_check": sum(len(r["unsupported_sentences"]) for r in rows),
        "refusal_recall_unanswerable": share(unanswerable, lambda r: r["refused"]),
        "false_refusal_rate_answerable": refused_answerable / len(answerable) if answerable else None,
        "refusal_decision_accuracy": correct_decisions / len(rows) if rows else None,
        "refusal_reasons": {
            reason: sum(1 for r in rows if r["refusal_reason"] == reason)
            for reason in {r["refusal_reason"] for r in rows if r["refusal_reason"]}
        },
        "avg_latency_s": statistics.mean(r["latency_s"] for r in rows) if rows else None,
    }


def spot_check_sheet(rows: list[dict], n: int = 10, seed: int = 7) -> str:
    candidates = [r for r in rows if r["answerable"]]
    sample = random.Random(seed).sample(candidates, min(n, len(candidates)))
    lines = [
        "# Manual spot-check of the LLM judge",
        "",
        "Score each answer yourself with the same 0-3 rubric, then compare with the judge.",
        "",
        "| id | question | reference | system answer | judge | human | agree? |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for r in sample:
        answer = r["answer_text"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['id']} | {r['question']} | {r['reference_answer']} | {answer} | "
            f"{r['correctness']['score'] if r.get('correctness') else '-'} |  |  |"
        )
    return "\n".join(lines) + "\n"


def run_phase(name: str, cases: list[dict], done: dict, path: Path, work, pause: float) -> int:
    """Process every case not already in ``done``; append each result to ``path``."""
    todo = [c for c in cases if c["id"] not in done]
    print(f"\n[{name}] {len(cases) - len(todo)} already done, {len(todo)} to go -> {path.name}", flush=True)
    failures = 0
    for index, case in enumerate(todo, start=1):
        try:
            row = work(case)
        except Exception as exc:  # keep going; the case is retried on the next run
            failures += 1
            print(f"  [{index}/{len(todo)}] {case['id']} FAILED: {exc.__class__.__name__}: {str(exc)[:150]}",
                  flush=True)
            if "quota" in str(exc).lower() or "exhausted" in str(exc).lower():
                print("  API quota exhausted - stopping. Re-run the same command later to resume.", flush=True)
                break
            continue
        append_checkpoint(path, row)
        done[case["id"]] = row
        print(f"  [{index}/{len(todo)}] {case['id']} {describe(row)}", flush=True)
        if pause:
            time.sleep(pause)
    return failures


def describe(row: dict) -> str:
    parts = [f"rank={row['rank']}", "REFUSED" if row["refused"] else "answered"]
    if row.get("correctness"):
        parts.append(f"score={row['correctness']['score']}")
    faith = (row.get("faithfulness") or {}).get("score")
    if faith is not None:
        parts.append(f"faith={faith:.2f}")
    parts.append(f"{row['latency_s']:.1f}s")
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True, help="run name; also the checkpoint file name")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--phase", default="all", choices=["generate", "judge", "all"])
    parser.add_argument("--fresh", action="store_true", help="delete this label's checkpoints and start over")
    parser.add_argument("--ids", nargs="*", help="only these question ids (run in batches)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--provider", default=None, help="generator LLM provider (default: LLM_PROVIDER)")
    parser.add_argument("--model", default=None, help="generator model (default: LLM_MODEL)")
    parser.add_argument("--judge-provider", default=None, help="default: same as --provider")
    parser.add_argument("--judge-model", default=None, help="default: same as --model")
    parser.add_argument("--naive", action="store_true",
                        help="original app: default LangChain prompt, top-3 context, no guardrails")
    parser.add_argument("--chunking", default="structured", choices=["fixed", "structured"])
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid"])
    parser.add_argument("--reranker", default=None)
    parser.add_argument("--embedding", default="all-MiniLM-L6-v2")
    parser.add_argument("--min-vector-score", type=float, default=0.30)
    parser.add_argument("--min-rerank-score", type=float, default=None)
    parser.add_argument("--citation-check", default="lexical", choices=["lexical", "llm", "off"])
    parser.add_argument("--citation-action", default="drop", choices=["drop", "flag"])
    parser.add_argument("--sleep", type=float, default=None,
                        help="pause after each question (default 4s for Gemini free tier, else 0)")
    args = parser.parse_args()

    if args.reranker == "bge-reranker-base":
        print("Warning: bge-reranker-base needs ~1.5 GB RAM and ~10 s/query on CPU; "
              "ms-marco-MiniLM is the light option.", flush=True)

    provider = (args.provider or os.getenv("LLM_PROVIDER", "ollama")).lower()
    pause = args.sleep if args.sleep is not None else (4.0 if provider == "gemini" else 0.0)

    cases = load_eval_set(Path(args.dataset))
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    cases = cases[: args.limit]
    by_id = {c["id"]: c for c in cases}

    answers_path = RESULTS_DIR / f"e2e_{args.label}.answers.jsonl"
    judged_path = RESULTS_DIR / f"e2e_{args.label}.judged.jsonl"
    if args.fresh:
        answers_path.unlink(missing_ok=True)
        judged_path.unlink(missing_ok=True)
    answers = read_checkpoint(answers_path)
    judged = read_checkpoint(judged_path)
    failures = 0

    if args.phase in {"generate", "all"}:
        llm = get_llm(provider=args.provider, model=args.model)
        pipeline = build_pipeline(args, llm)
        failures += run_phase("generate", cases, answers, answers_path,
                              lambda case: generate_row(pipeline, case), pause)
        del pipeline  # free the embedding model before judging

    if args.phase in {"judge", "all"}:
        judge = get_llm(provider=args.judge_provider or args.provider, model=args.judge_model or args.model)
        ready = [by_id[i] for i in answers if i in by_id]
        failures += run_phase("judge", ready, judged, judged_path,
                              lambda case: judge_row(judge, answers[case["id"]], case, cited=not args.naive),
                              pause)

    rows = [judged[c["id"]] for c in cases if c["id"] in judged]
    missing = len(cases) - len(rows)
    if not rows:
        print("\nNo judged rows yet.")
        return 1 if failures else 0

    summary = summarize(rows)
    config = {k: v for k, v in vars(args).items() if k not in {"label", "fresh", "ids", "limit", "phase"}}
    write_json(RESULTS_DIR / f"e2e_{args.label}.json",
               {"config": config, "complete": missing == 0, "summary": summary, "rows": rows})
    (RESULTS_DIR / f"spot_check_{args.label}.md").write_text(spot_check_sheet(rows), encoding="utf-8")

    print(f"\nSummary over {len(rows)} judged questions" + (f" ({missing} still missing)" if missing else ""))
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    print(f"\nWrote {RESULTS_DIR / f'e2e_{args.label}.json'}")
    return 1 if failures or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
