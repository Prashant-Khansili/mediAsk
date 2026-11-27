# MediAsk evaluation results

Evaluation set: `evaluation/eval_set.jsonl`, 50 questions written from the indexed
Acetaminophen Injection label (40 answerable, 10 unanswerable). Generator and judge:
`gemini-3.5-flash-lite`. Run on a CPU-only laptop; only the embedding model ran locally.

## Before / after (end to end)

| Metric | Original app | Final pipeline |
|---|---:|---:|
| Retrieval hit@5 | 45% | **97.5%** |
| Retrieval MRR@5 | 0.45 | **0.84** |
| Answers judged correct (score ≥ 2 of 3) | 55% | **97.5%** |
| Mean correctness (0–3) | 1.50 | **2.75** |
| Faithfulness (answer sentences supported by cited chunk) | 0.976\* | **1.00** |
| Wrong "not found" on answerable questions | 35% | **2.5%** (1 of 40) |
| Unanswerable questions refused | 10 / 10 | 10 / 10 |
| Correct answer-or-refuse decisions | 72% | **98%** |
| Average latency per question | 9.8 s | 5.9 s |

\* The original app gives no citations, so its faithfulness is judged against any
retrieved chunk, a looser test than the per-citation check used for the final pipeline.

- **Original app** (`--naive`): 500-character chunks, six duplicate copies of the PDF
  indexed, MiniLM vector search, LangChain's default prompt over the top 3 chunks,
  no guardrails.
- **Final pipeline**: duplicate files skipped, heading-aware chunks with tables kept whole,
  BM25 + e5-small hybrid search (reciprocal rank fusion), no re-ranker, guarded
  prompt with citations, citation check, and refusal threshold 0.80.

Files: `e2e_baseline.json`, `e2e_final.json` (per-question rows), `spot_check_final.md`.

## Retrieval ablation (one change at a time)

See `retrieval_experiments.md` and `retrieval_experiments_hybrid_e5_small.md`.

| Step | hit@5 | MRR@5 | ms/query |
|---|---:|---:|---:|
| Original (duplicates, fixed chunks, MiniLM) | 45% | 0.450 | 18 |
| + skip duplicate files | 80% | 0.609 | 17 |
| + heading-aware chunking | 90% | 0.693 | 19 |
| + BM25 hybrid search (RRF) | 92.5% | 0.775 | 28 |
| + ms-marco-MiniLM re-ranker | 92.5% | 0.810 | 1,786 |
| + bge-reranker-base instead | 95% | 0.850 | 10,703 |
| Hybrid with e5-small, no re-ranker (**chosen**) | **97.5%** | 0.838 | **42** |

The re-rankers were not shipped: e5-small with hybrid search matched them within one
question at about 1/250th of the latency on CPU.

## Guardrails

- **Prompt injection** (`injection_test.json`): 5 poisoned documents; 4 were retrieved into
  the context. 0 of 4 succeeded against the guarded pipeline. The original prompt also
  resisted all 4 with this model, so this test shows the defence holds, but **not** that
  the delimiters were what stopped these particular attacks. A weaker or older model is
  needed to measure the defence's added value.
- **Citation check** dropped 1 unsupported sentence across 50 answers; raw faithfulness was
  already 1.00 with this model.
- **Similarity threshold** cannot separate unanswerable near-miss questions: they score as
  high as answerable ones because they ask about the same drug. All 11 refusals in the final run came
  from the model replying `NOT_FOUND`; the threshold only catches clearly off-topic questions.

## Caveats

- The same model generated and judged the answers, which can inflate scores. Fill in the
  human column of `spot_check_final.md` (10 answers) to check judge agreement.
- 40 answerable questions from a single 16-page label: one question = 2.5 points, so treat
  small differences in the ablation as noise.
- The questions were written from the document, which favours keyword search (BM25 alone
  reached 95% hit@5).
- The one miss, `q16` ("Who should not receive Acetaminophen Injection?"), shows the
  reverse. Vector search ranks the contraindications chunk 2nd, but BM25 ranks it 37th–39th
  because the label says "contraindicated" and the question shares none of its words.
  Fusing both rankings with equal weight puts the chunk 13th, outside the top 5. Real users
  phrase questions like `q16`, so a held-out set of questions in everyday wording is the
  next thing to measure before tuning the fusion weights or adding synonym matching.
