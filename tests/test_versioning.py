from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worklib import versioning


class VersioningTest(unittest.TestCase):
    def test_latest_manifest_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "v1" / "_state").mkdir(parents=True)
            (root / "v2" / "_state").mkdir(parents=True)
            manifest = root / "v2" / "_state" / "library.json"
            manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(versioning.latest_version(root), "v2")
            self.assertEqual(versioning.latest_manifest_path(root), manifest)

    def test_archive_version_folder_deletes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1 = root / "v1"
            v1.mkdir()
            (v1 / "foo.txt").write_text("hello", encoding="utf-8")

            archive = versioning.archive_version_folder(v1)
            self.assertTrue(archive.exists())
            self.assertFalse(v1.exists())

    def test_resolve_manifest_path_prefers_latest_when_no_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "v3" / "_state" / "library.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")

            with patch("worklib.versioning.get_biblioteca_root", return_value=root):
                resolved = versioning.resolve_manifest_path(None)

            self.assertEqual(resolved, manifest)


if __name__ == "__main__":
    unittest.main()
