| Step | Change | Chunks | hit@1 | hit@5 | MRR@5 | unique/top5 | query ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline` | Original: 500-char chunks, vector only, MiniLM, duplicate PDFs indexed | 654 | 45% | 45% | 0.450 | 1.0 | 18 |
| `dedupe` | + skip byte-identical files | 109 | 45% | 80% | 0.609 | 5.0 | 17 |
| `structured` | + heading-aware chunks, tables whole, drug/section/page metadata | 61 | 57% | 90% | 0.693 | 5.0 | 19 |
| `bm25_only` | (reference) keyword search alone | 61 | 75% | 95% | 0.820 | 5.0 | 24 |
| `hybrid` | + BM25 + vector fused with RRF | 61 | 65% | 92% | 0.775 | 5.0 | 28 |
| `rerank_minilm` | + re-rank top 20 with ms-marco-MiniLM-L-6 cross-encoder | 61 | 72% | 92% | 0.810 | 5.0 | 1786 |
| `rerank_bge` | + re-rank top 20 with bge-reranker-base (instead of MiniLM) | 61 | 78% | 95% | 0.850 | 5.0 | 10703 |
| `emb_bge_small_vec` | vector only, bge-small-en-v1.5 | 61 | 70% | 92% | 0.780 | 5.0 | 45 |
| `emb_bge_base_vec` | vector only, bge-base-en-v1.5 | 61 | 68% | 92% | 0.767 | 5.0 | 111 |
| `emb_e5_small_vec` | vector only, e5-small-v2 | 61 | 72% | 98% | 0.826 | 5.0 | 42 |
| `emb_e5_base_vec` | vector only, e5-base-v2 | 61 | 70% | 98% | 0.811 | 5.0 | 108 |
| `full_bge_small` | hybrid + rerank, bge-small | 61 | 72% | 95% | 0.815 | 5.0 | 1506 |
| `full_bge_base` | hybrid + rerank, bge-base | 61 | 72% | 92% | 0.810 | 5.0 | 1563 |
| `full_e5_base` | hybrid + rerank, e5-base | 61 | 72% | 95% | 0.817 | 5.0 | 1570 |
