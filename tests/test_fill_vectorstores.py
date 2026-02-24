from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from library_ops import fill_vectorstores


class _FileObj:
    def __init__(self, file_id: str) -> None:
        self.id = file_id


class _VectorStoreFiles:
    def __init__(self) -> None:
        self.by_vs = {"vs_existing": ["file_existing"]}

    def list(self, vector_store_id: str):
        return [_FileObj(file_id) for file_id in self.by_vs.get(vector_store_id, [])]

    def create(self, *, vector_store_id: str, file_id: str):
        self.by_vs.setdefault(vector_store_id, []).append(file_id)


class _VectorStores:
    def __init__(self) -> None:
        self.files = _VectorStoreFiles()


class _Client:
    def __init__(self) -> None:
        self.vector_stores = _VectorStores()


class FillVectorstoresTest(unittest.TestCase):
    def test_uploads_missing_file_ids_and_attaches_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "doc1.txt"
            p2 = Path(tmp) / "doc2.txt"
            p1.write_text("uno", encoding="utf-8")
            p2.write_text("dos", encoding="utf-8")
            manifest_path = Path(tmp) / "library.json"
            manifest = {
                "categories": {"cat": {"name": "cat", "vector_store_id": "vs_existing"}},
                "docs": {
                    "d1": {"doc_id": "d1", "category": "cat", "abs_path": str(p1), "openai_file_id": "file_existing"},
                    "d2": {"doc_id": "d2", "category": "cat", "abs_path": str(p2), "openai_file_id": ""},
                },
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            client = _Client()
            with patch("library_ops.fill_vectorstores.get_client_from_env", return_value=client), patch(
                "library_ops.fill_vectorstores.upload_file", return_value="file_new"
            ):
                rc = fill_vectorstores.run(manifest=manifest_path, debug=False)

            self.assertEqual(rc, 0)
            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(updated["docs"]["d2"]["openai_file_id"], "file_new")
            self.assertEqual(client.vector_stores.files.by_vs["vs_existing"], ["file_existing", "file_new"])


if __name__ == "__main__":
    unittest.main()
