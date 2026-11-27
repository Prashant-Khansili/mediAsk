| Step | Change | Chunks | hit@1 | hit@5 | MRR@5 | unique/top5 | query ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline` | Original: 500-char chunks, vector only, MiniLM, duplicate PDFs indexed | 654 | 45% | 45% | 0.450 | 1.0 | 19 |
| `dedupe` | + skip byte-identical files | 109 | 45% | 80% | 0.609 | 5.0 | 16 |
| `structured` | + heading-aware chunks, tables whole, drug/section/page metadata | 61 | 57% | 90% | 0.693 | 5.0 | 18 |
| `bm25_only` | (reference) keyword search alone | 61 | 75% | 95% | 0.820 | 5.0 | 18 |
| `hybrid` | + BM25 + vector fused with RRF | 61 | 65% | 92% | 0.775 | 5.0 | 20 |
