import unittest

from langchain_core.documents import Document

from rag.chunking import chunk_documents
from rag.retrieval import BM25, reciprocal_rank_fusion, tokenize

LABEL_PAGES = [
    Document(
        page_content=(
            "2 DOSAGE AND ADMINISTRATION\n"
            "2.2 Recommended Dosage: Adults\n"
            "Adults weighing 50 kg and over: 1,000 mg every 6 hours.\n"
            "Table 1. Dosing for Adults\n"
            "Age group Dose every 6 hours Maximum daily dose\n"
            "Adults 1,000 mg 4,000 mg\n"
            "Running footer text\n"
        ),
        metadata={"source": "data/label.pdf", "page": 0},
    ),
    Document(
        page_content="7 DRUG INTERACTIONS\n7.2 Anticoagulants\nWarfarin INR may increase.\nRunning footer text\n",
        metadata={"source": "data/label.pdf", "page": 1},
    ),
    Document(
        page_content="16 HOW SUPPLIED\nStore at 20°C to 25°C.\nRunning footer text\n",
        metadata={"source": "data/label.pdf", "page": 2},
    ),
]


class RetrievalPrimitiveTests(unittest.TestCase):
    def test_tokenize_joins_thousands_and_drops_stopwords(self):
        self.assertEqual(tokenize("What is the dose of 4,000 mg?"), ["dose", "4000", "mg"])

    def test_bm25_prefers_exact_rare_terms(self):
        bm25 = BM25(["acetaminophen dosing for adults", "warfarin interaction raises INR", "storage temperature"])
        self.assertEqual(bm25.top("warfarin", 1)[0][0], 1)
        self.assertEqual(bm25.top("nothingmatches", 3), [])

    def test_rrf_rewards_documents_ranked_high_in_both_lists(self):
        fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
        self.assertEqual(fused[0][0], 1)
        self.assertEqual({doc_id for doc_id, _ in fused}, {1, 2, 3, 4})


class StructuredChunkingTests(unittest.TestCase):
    def setUp(self):
        self.chunks = chunk_documents(LABEL_PAGES, "structured")

    def test_chunks_follow_headings_and_carry_metadata(self):
        sections = [c.metadata["section"] for c in self.chunks]
        self.assertIn("2.2 Recommended Dosage: Adults", sections)
        self.assertIn("7.2 Anticoagulants", sections)
        warfarin = next(c for c in self.chunks if "Warfarin" in c.page_content)
        self.assertEqual(warfarin.metadata["page"], 2)
        self.assertTrue(warfarin.page_content.startswith("[Label | 7.2 Anticoagulants]"))

    def test_table_stays_in_one_chunk_with_its_section(self):
        table = [c for c in self.chunks if "Table 1." in c.page_content]
        self.assertEqual(len(table), 1)
        self.assertIn("Adults 1,000 mg 4,000 mg", table[0].page_content)

    def test_repeated_page_footer_is_removed(self):
        self.assertFalse(any("Running footer" in c.page_content for c in self.chunks))

    def test_markdown_headings_split_sections(self):
        doc = Document(page_content="# Ibuprofen\n## Dosage\n200 mg.\n## Warnings\nGI bleeding.",
                       metadata={"source": "ibuprofen.md"})
        chunks = chunk_documents([doc], "structured")
        self.assertEqual([c.metadata["section"] for c in chunks], ["Dosage", "Warnings"])
        self.assertEqual(chunks[0].metadata["drug"], "Ibuprofen")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_documents(LABEL_PAGES, "semantic")


if __name__ == "__main__":
    unittest.main()
