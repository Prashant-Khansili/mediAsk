"""Build a FAISS index from supported medical documents in the data directory."""

from pathlib import Path
import os
import tempfile
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "rag" / "faiss_index"
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".docx"}


def load_docs(doc_path: str | Path = DEFAULT_DATA_DIR):
    """Load supported medical files in ``doc_path`` in filename order."""
    docs = []
    for source_path in sorted(Path(doc_path).iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

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


def chunk_docs(docs: Iterable):
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def create_vectorstore(
    chunks: list,
    persist_dir: str | Path = DEFAULT_INDEX_DIR,
):
    """Embed chunks and persist the FAISS index."""
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(persist_path))
    print(f"Saved index to {persist_path}")


def build_index(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
):
    """Rebuild the index atomically from every supported file in ``data_dir``."""
    docs = load_docs(data_dir)
    if not docs:
        raise ValueError(
            f"No supported medical documents found in {Path(data_dir)} "
            f"({', '.join(sorted(SUPPORTED_EXTENSIONS))})"
        )

    chunks = chunk_docs(docs)
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=index_path.parent) as temp_dir:
        create_vectorstore(chunks, temp_dir)
        for index_file in Path(temp_dir).iterdir():
            os.replace(index_file, index_path / index_file.name)

    return len(docs), len(chunks)


if __name__ == "__main__":
    build_index()
