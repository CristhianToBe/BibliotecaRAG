from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from library_ops import relink_openai_files


class _Page:
    def __init__(self, data, next_page=None):
        self.data = data
        self._next_page = next_page

    def has_next_page(self):
        return self._next_page is not None

    def get_next_page(self):
        return self._next_page


class _FilesAPI:
    def __init__(self, first_page):
        self._first_page = first_page

    def list(self, **kwargs):
        return self._first_page


class _Client:
    def __init__(self, first_page):
        self.files = _FilesAPI(first_page)


class RelinkOpenAIFilesTest(unittest.TestCase):
    def test_relinks_missing_ids_using_filename_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "library.json"
            manifest = {
                "categories": {},
                "docs": {
                    "d1": {"doc_id": "d1", "filename": "a.pdf", "size_bytes": 100, "openai_file_id": ""},
                    "d2": {"doc_id": "d2", "filename": "b.pdf", "size_bytes": 222, "openai_file_id": ""},
                    "d3": {"doc_id": "d3", "filename": "c.pdf", "openai_file_id": "already"},
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            page2 = _Page([SimpleNamespace(id="file_b", filename="b.pdf", bytes=222, created_at=2)])
            page1 = _Page([
                SimpleNamespace(id="file_a_old", filename="a.pdf", bytes=50, created_at=1),
                SimpleNamespace(id="file_a_new", filename="a.pdf", bytes=100, created_at=3),
            ], next_page=page2)
            client = _Client(page1)

            with patch("library_ops.relink_openai_files.get_client_from_env", return_value=client):
                rc = relink_openai_files.run(manifest=manifest_path, also_match_by_bytes=True)

            self.assertEqual(rc, 0)
            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["docs"]["d1"]["openai_file_id"], "file_a_new")
            self.assertEqual(updated["docs"]["d2"]["openai_file_id"], "file_b")
            self.assertEqual(updated["docs"]["d3"]["openai_file_id"], "already")


if __name__ == "__main__":
    unittest.main()
