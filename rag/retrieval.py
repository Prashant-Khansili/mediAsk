"""Retrieval: dense (FAISS), BM25 keyword search, reciprocal rank fusion, re-ranking."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

RERANKER_MODELS = {
    "ms-marco-MiniLM": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}

_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i in is it its may
    of on or should the to was were what when which who why will with this that these
    those than there their been being if into""".split()
)


def tokenize(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())  # 4,000 -> 4000
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        frequency = Counter(term for doc in self.docs for term in doc)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in frequency.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        results = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
                score += self.idf[term] * tf * (self.k1 + 1) / norm
            results.append(score)
        return results

    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda item: item[1], reverse=True)
        return [(index, score) for index, score in ranked[:k] if score > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids: score(d) = sum 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


@dataclass
class RetrievalConfig:
    top_k: int = 5
    mode: str = "hybrid"  # "vector" | "bm25" | "hybrid"
    candidates: int = 20  # depth of each list before fusion / re-ranking
    reranker: str | None = None  # key of RERANKER_MODELS or a HF model id

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        reranker = os.getenv("RERANKER", "").strip() or None
        if reranker and reranker.lower() in {"none", "off", "false", "0"}:
            reranker = None
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            mode=os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower(),
            candidates=int(os.getenv("RETRIEVAL_CANDIDATES", "20")),
            reranker=reranker,
        )


@dataclass
class RetrievedChunk:
    document: Document
    vector_score: float | None = None  # cosine similarity (None if not in dense list)
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None  # sigmoid of cross-encoder logit, 0..1
    extras: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    best_vector_score: float  # highest cosine similarity among dense candidates
    best_rerank_score: float | None

    @property
    def documents(self) -> list[Document]:
        return [chunk.document for chunk in self.chunks]


class HybridRetriever:
    """Dense + BM25 retrieval with RRF fusion and optional cross-encoder re-ranking."""

    def __init__(self, vectorstore, config: RetrievalConfig | None = None):
        self.vectorstore = vectorstore
        self.config = config or RetrievalConfig()
        store = vectorstore.docstore._dict
        self.ids = [vectorstore.index_to_docstore_id[i] for i in range(len(vectorstore.index_to_docstore_id))]
        self.docs = [store[doc_id] for doc_id in self.ids]
        self.position = {doc_id: index for index, doc_id in enumerate(self.ids)}
        self.bm25 = BM25([doc.page_content for doc in self.docs])

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        embedding = self.vectorstore.embedding_function.embed_query(query)
        scores, indices = self.vectorstore.index.search(
            np.array([embedding], dtype="float32"), min(k, len(self.docs))
        )
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def retrieve(self, query: str) -> RetrievalResult:
        cfg = self.config
        depth = max(cfg.candidates, cfg.top_k)
        dense = self._dense(query, depth) if cfg.mode in {"vector", "hybrid"} else []
        keyword = self.bm25.top(query, depth) if cfg.mode in {"bm25", "hybrid"} else []
        # Always know the best dense similarity: it drives the refusal threshold.
        probe = dense or self._dense(query, 1)
        best_vector = probe[0][1] if probe else 0.0

        dense_scores = dict(dense)
        keyword_scores = dict(keyword)
        if cfg.mode == "vector":
            ranked = dense
        elif cfg.mode == "bm25":
            ranked = keyword
        else:
            ranked = reciprocal_rank_fusion([[i for i, _ in dense], [i for i, _ in keyword]])
        ranked = ranked[:depth]

        chunks = [
            RetrievedChunk(
                document=self.docs[i],
                vector_score=dense_scores.get(i),
                bm25_score=keyword_scores.get(i),
                fused_score=score,
            )
            for i, score in ranked
        ]

        best_rerank = None
        if cfg.reranker and chunks:
            model = _load_cross_encoder(RERANKER_MODELS.get(cfg.reranker, cfg.reranker))
            raw = np.asarray(model.predict([(query, c.document.page_content) for c in chunks]), dtype=float)
            if raw.min() < 0 or raw.max() > 1:  # some versions return logits
                raw = 1.0 / (1.0 + np.exp(-raw))
            for chunk, score in zip(chunks, raw):
                chunk.rerank_score = float(score)
            chunks.sort(key=lambda c: c.rerank_score, reverse=True)
            best_rerank = chunks[0].rerank_score

        return RetrievalResult(
            chunks=chunks[: cfg.top_k],
            best_vector_score=best_vector,
            best_rerank_score=best_rerank,
        )
