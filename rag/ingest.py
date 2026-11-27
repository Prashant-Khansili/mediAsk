"""Build a FAISS index from supported medical documents in the data directory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from rag.chunking import chunk_documents
from rag.embeddings import DEFAULT_EMBEDDING_MODEL, LEGACY_EMBEDDING_MODEL, PrefixedSentenceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "faiss_index"
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}
INDEX_CONFIG_FILE = "index_config.json"
DEFAULT_CHUNKING = "structured"


def load_docs(doc_path: str | Path = DEFAULT_DATA_DIR, dedupe: bool = True):
    """Load supported medical files in ``doc_path`` in filename order.

    Byte-identical files (e.g. the same PDF uploaded twice) are loaded once so
    duplicates cannot crowd out other results at retrieval time.
    """
    docs = []
    seen_hashes: set[str] = set()
    for source_path in sorted(Path(doc_path).iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if dedupe and digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        if source_path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(source_path)).load())
        elif source_path.suffix.lower() == ".md":
            docs.append(
                Document(
                    page_content=source_path.read_text(encoding="utf-8"),
                    metadata={"source": str(source_path)},
                )
            )
        else:
            docs.extend(_load_docx(source_path))
    return docs


def _load_docx(source_path: Path) -> list[Document]:
    """Extract paragraphs from a DOCX file while preserving its source."""
    from docx import Document as DocxDocument

    document = DocxDocument(str(source_path))
    content = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )
    if not content:
        return []
    return [Document(page_content=content, metadata={"source": str(source_path)})]


def chunk_docs(docs: Iterable, strategy: str | None = None):
    """Split documents into chunks using the configured strategy."""
    strategy = strategy or os.getenv("CHUNKING_STRATEGY", DEFAULT_CHUNKING)
    return chunk_documents(docs, strategy)


def make_vectorstore(chunks: list, embedding_model: str | None = None):
    """Embed chunks into an in-memory FAISS store (cosine similarity)."""
    embeddings = PrefixedSentenceEmbeddings(embedding_model)
    return FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def create_vectorstore(
    chunks: list,
    persist_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: str | None = None,
    chunking: str | None = None,
):
    """Embed chunks and persist the FAISS index plus the settings used to build it."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    vectorstore = make_vectorstore(chunks, embedding_model)
    vectorstore.save_local(str(persist_path))
    config = {
        "embedding_model": embedding_model
        or os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "chunking": chunking or os.getenv("CHUNKING_STRATEGY", DEFAULT_CHUNKING),
        "chunks": len(chunks),
    }
    (persist_path / INDEX_CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved index to {persist_path}")


def read_index_config(index_dir: str | Path = DEFAULT_INDEX_DIR) -> dict:
    path = Path(index_dir) / INDEX_CONFIG_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"embedding_model": LEGACY_EMBEDDING_MODEL, "chunking": "fixed"}


def build_index(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    embedding_model: str | None = None,
    chunking: str | None = None,
):
    """Rebuild the index atomically from every supported file in ``data_dir``."""
    docs = load_docs(data_dir)
    if not docs:
        raise ValueError(
            f"No supported medical documents found in {Path(data_dir)} "
            f"({', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    chunks = chunk_docs(docs, chunking)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=index_path.parent) as temp_dir:
        create_vectorstore(chunks, temp_dir, embedding_model, chunking)
        for index_file in Path(temp_dir).iterdir():
            os.replace(index_file, index_path / index_file.name)

    return len(docs), len(chunks)


if __name__ == "__main__":
    build_index()
