| Step | Change | Chunks | hit@1 | hit@5 | MRR@5 | unique/top5 | query ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `rerank_bge` | + re-rank top 20 with bge-reranker-base (instead of MiniLM) | 61 | 78% | 95% | 0.850 | 5.0 | 8271 |
