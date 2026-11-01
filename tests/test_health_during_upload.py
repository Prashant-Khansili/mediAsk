import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class HealthDuringUploadTests(unittest.TestCase):
    def test_upload_runs_indexing_without_blocking_request_handling(self):
        client = TestClient(app)
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("app.main.DEFAULT_DATA_DIR", Path(directory)),
                patch("app.main.build_index", return_value=(1, 1)) as rebuild,
            ):
                response = client.post(
                    "/upload",
                    files={"file": ("notes.md", b"medical notes", "text/markdown")},
                )

        self.assertEqual(response.status_code, 200)
        rebuild.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
