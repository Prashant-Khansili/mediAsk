import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class UploadTests(unittest.TestCase):
    def test_upload_rejects_unsupported_files(self):
        client = TestClient(app)

        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"not a medical PDF", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(".md, .pdf, and .docx", response.json()["detail"])

    def test_upload_stores_supported_files_and_rebuilds_index(self):
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with (
                patch("app.main.DEFAULT_DATA_DIR", data_dir),
                patch("app.main.build_index", return_value=(2, 4)) as rebuild,
            ):
                for filename, content, content_type in (
                    ("reference.pdf", b"%PDF-1.4 test", "application/pdf"),
                    ("notes.md", b"# Medication notes", "text/markdown"),
                    (
                        "reference.docx",
                        b"PK\x03\x04 test",
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                ):
                    response = client.post(
                        "/upload",
                        files={"file": (filename, content, content_type)},
                    )

                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["chunks_indexed"], 4)
                    self.assertEqual(
                        len(list(data_dir.glob(f"*_{filename}"))),
                        1,
                    )
            self.assertEqual(rebuild.call_count, 3)


if __name__ == "__main__":
    unittest.main()
