import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag.ingest import chunk_docs, create_vectorstore, load_docs


class IngestTests(unittest.TestCase):
    def test_load_docs_ignores_unsupported_files(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            (data_dir / "notes.txt").write_text("not a PDF", encoding="utf-8")

            self.assertEqual(
                load_docs(data_dir),
                [],
            )

    def test_load_docs_supports_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            markdown_path = data_dir / "notes.md"
            markdown_path.write_text("# Medication notes", encoding="utf-8")

            docs = load_docs(data_dir)

            self.assertEqual(len(docs), 1)
            self.assertEqual(docs[0].page_content, "# Medication notes")
            self.assertEqual(docs[0].metadata["source"], str(markdown_path))

    def test_chunk_docs_preserves_document_content(self):
        chunks = chunk_docs(
            [Document(page_content="Medication information " * 60)]
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(
            all(chunk.page_content for chunk in chunks)
        )

    def test_create_vectorstore_creates_index_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            index_dir = Path(directory) / "nested" / "index"

            class FakeEmbeddings:
                pass

            import rag.ingest as ingest
            original_embeddings = ingest.HuggingFaceEmbeddings
            original_faiss = ingest.FAISS
            try:
                ingest.HuggingFaceEmbeddings = (
                    lambda model_name: FakeEmbeddings()
                )

                class FakeVectorStore:
                    def save_local(self, path):
                        Path(path).mkdir(parents=True, exist_ok=True)

                ingest.FAISS = type(
                    "FakeFAISS",
                    (),
                    {
                        "from_documents": staticmethod(
                            lambda chunks, embeddings: FakeVectorStore()
                        )
                    },
                )
                create_vectorstore(
                    [Document(page_content="content")],
                    index_dir,
                )
            finally:
                ingest.HuggingFaceEmbeddings = original_embeddings
                ingest.FAISS = original_faiss

            self.assertTrue(index_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
