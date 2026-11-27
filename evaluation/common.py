"""Shared helpers for the MediAsk evaluation scripts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_DATASET = EVAL_DIR / "eval_set.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


# LangChain's default RetrievalQA "stuff" prompt: what the original app used.
NAIVE_PROMPT = """Use the following pieces of context to answer the question at the end. \
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""

_NAIVE_REFUSAL = re.compile(
    r"(i don't know|i do not know|not (?:mentioned|provided|available|specified|stated|included|contain)"
    r"|no information|does not (?:contain|provide|mention|specify)|doesn't (?:contain|provide|mention|specify)"
    r"|cannot answer|can't answer|unable to answer)",
    re.IGNORECASE,
)


def naive_refused(answer: str) -> bool:
    return bool(_NAIVE_REFUSAL.search(answer))


def load_eval_set(path: Path = DEFAULT_DATASET) -> list[dict]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        for key in ("id", "question", "answerable", "reference_answer"):
            if key not in case:
                raise ValueError(f"{path}:{number} is missing '{key}'")
        if case["answerable"] and not case.get("evidence"):
            raise ValueError(f"{path}:{number} answerable case needs an 'evidence' span")
        cases.append(case)
    return cases


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk_text: str, evidence: str) -> bool:
    """A chunk is the 'correct chunk' if it contains the case's evidence span."""
    return normalize(evidence) in normalize(chunk_text)


def first_relevant_rank(chunk_texts: list[str], evidence: str) -> int | None:
    for rank, text in enumerate(chunk_texts, start=1):
        if is_relevant(text, evidence):
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], k: int = 5) -> dict:
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n,
        f"mrr@{k}": sum(1 / r for r in ranks if r is not None and r <= k) / n,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
