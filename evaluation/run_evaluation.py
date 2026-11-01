"""Run a lightweight evaluation against the MediAsk /ask endpoint."""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def load_cases(path: Path) -> list[dict]:
    content = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

    if not isinstance(parsed, list):
        raise ValueError("Evaluation data must be a JSON array or JSONL file.")

    cases = []
    for index, case in enumerate(parsed, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Evaluation case {index} must be a JSON object.")
        for field in ("question", "expected_sources", "reference_answer"):
            if field not in case:
                raise ValueError(f"Missing '{field}' in evaluation case {index}")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def answer_overlap(answer: str, reference_answer: str) -> float:
    reference_tokens = tokens(reference_answer)
    if not reference_tokens:
        return 0.0
    return len(tokens(answer) & reference_tokens) / len(reference_tokens)


def request_answer(api_url: str, question: str, timeout: float) -> tuple[dict, float]:
    payload = json.dumps({"query": question}).encode("utf-8")
    request = Request(
        f"{api_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    return result, time.perf_counter() - started


def source_match(expected_sources: list[str], sources: list[str]) -> bool:
    source_text = "\n".join(sources).lower()
    return bool(expected_sources) and all(
        expected.lower() in source_text for expected in expected_sources
    )


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run(cases: list[dict], api_url: str, timeout: float) -> int:
    overlaps = []
    latencies = []
    source_matches = 0
    failures = 0

    for index, case in enumerate(cases, start=1):
        try:
            result, latency = request_answer(api_url, case["question"], timeout)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            overlaps.append(answer_overlap(answer, case["reference_answer"]))
            latencies.append(latency)
            source_matches += source_match(case["expected_sources"], sources)
            print(f"[{index}/{len(cases)}] {latency:.2f}s - {case['question']}")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            failures += 1
            print(f"[{index}/{len(cases)}] FAILED - {case['question']}: {exc}")

    completed = len(overlaps)
    print("\nEvaluation summary")
    print(f"Cases: {len(cases)}")
    print(f"Completed: {completed}")
    print(f"Failures: {failures}")
    if not completed:
        return 1
    print(f"Answer token overlap: {statistics.mean(overlaps):.2%}")
    print(f"Expected-source match: {source_matches / completed:.2%}")
    print(f"Average latency: {statistics.mean(latencies):.2f}s")
    print(f"P95 latency: {percentile(latencies, 0.95):.2f}s")
    print(
        "\nNote: /ask currently returns source text rather than source filenames. "
        "Expected-source match is meaningful only when filenames appear in that text."
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/questions.jsonl"),
        help="Path to the JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="MediAsk API base URL.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    return run(load_cases(args.dataset), args.api_url, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
