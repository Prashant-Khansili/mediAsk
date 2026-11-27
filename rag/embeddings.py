"""Sentence-transformer embeddings with the query/passage prefixes each model expects."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.embeddings import Embeddings

# Chosen by evaluation/retrieval_experiments.py: with hybrid search it reached
# hit@5 97.5% / MRR 0.838 on the 40 answerable questions at ~40 ms per query.
DEFAULT_EMBEDDING_MODEL = "e5-small"
LEGACY_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # indexes built before index_config.json existed

# Refusal threshold on the best cosine similarity, per model. Cosine scales differ
# a lot between models, so one global number does not transfer. Each value sits just
# below the lowest score of any answerable eval question, so it only rejects clearly
# off-topic questions (unanswerable near-misses score higher; the model's NOT_FOUND
# handles those).
DEFAULT_MIN_SCORES = {"all-MiniLM-L6-v2": 0.30, "e5-small": 0.80}

# Short aliases -> (HF model id, query prefix, passage prefix)
EMBEDDING_MODELS: dict[str, tuple[str, str, str]] = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": (
        "BAAI/bge-small-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "bge-base": (
        "BAAI/bge-base-en-v1.5",
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
}


def resolve_embedding_model(name: str | None = None) -> tuple[str, str, str]:
    name = (name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)).strip()
    return EMBEDDING_MODELS.get(name, (name, "", ""))


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


class PrefixedSentenceEmbeddings(Embeddings):
    """Normalised embeddings so inner product == cosine similarity."""

    def __init__(self, name: str | None = None):
        self.name = (name or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)).strip()
        self.model_id, self.query_prefix, self.passage_prefix = resolve_embedding_model(self.name)

    @property
    def model(self):
        return _load_model(self.model_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            [self.passage_prefix + text for text in texts],
            normalize_embeddings=True,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(self.query_prefix + text, normalize_embeddings=True)
        return vector.tolist()
