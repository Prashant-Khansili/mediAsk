import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate import append_checkpoint, parse_json, read_checkpoint, summarize


class CheckpointTests(unittest.TestCase):
    def test_resume_skips_half_written_line_after_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.answers.jsonl"
            append_checkpoint(path, {"id": "q01", "answer_text": "ok"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"id": "q02", "answer_te')  # process killed mid-write
            self.assertEqual(list(read_checkpoint(path)), ["q01"])
            append_checkpoint(path, {"id": "q02", "answer_text": "ok"})  # retried on the next run
            self.assertEqual(set(read_checkpoint(path)), {"q01", "q02"})

    def test_parse_json_ignores_text_after_the_object(self):
        text = 'Here you go:\n```json\n{"score": 3, "reason": "ok", "verdicts": []}\n```\n{"extra": 1}'
        self.assertEqual(parse_json(text)["score"], 3)

    def test_summary_counts_refusals_and_correctness(self):
        rows = [
            {"answerable": True, "rank": 1, "refused": False, "refusal_reason": None, "latency_s": 1.0,
             "correctness": {"score": 3}, "faithfulness": {"score": 1.0}, "unsupported_sentences": []},
            {"answerable": True, "rank": None, "refused": True, "refusal_reason": "model_not_found",
             "latency_s": 1.0, "correctness": {"score": 0}, "unsupported_sentences": []},
            {"answerable": False, "rank": None, "refused": True, "refusal_reason": "model_not_found",
             "latency_s": 1.0, "unsupported_sentences": []},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["hit@5"], 0.5)
        self.assertEqual(summary["refusal_recall_unanswerable"], 1.0)
        self.assertEqual(summary["false_refusal_rate_answerable"], 0.5)
        self.assertEqual(summary["correctness_mean_0_3"], 1.5)


if __name__ == "__main__":
    unittest.main()
