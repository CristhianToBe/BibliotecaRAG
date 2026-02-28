from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from library_ops import orchestrate


class OrchestrateEnforcementTest(unittest.TestCase):
    def _mk_version(self, root: Path, version: str, manifest_data: dict) -> Path:
        manifest = root / version / "_state" / "library.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    def test_dry_run_prints_plan_and_skips_mutating_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            self._mk_version(lib, "v1", {"categories": {}, "docs": {}})

            calls = {"vs": 0, "kw": 0, "del": 0, "smoke": 0, "archive": 0}

            with patch("library_ops.orchestrate.get_biblioteca_root", return_value=lib), \
                patch("library_ops.orchestrate.rebuild_versioned.run", return_value=0), \
                patch("library_ops.orchestrate.fill_vectorstores.run", side_effect=lambda **_: calls.__setitem__("vs", calls["vs"] + 1) or 0), \
                patch("library_ops.orchestrate.fill_keywords.run", side_effect=lambda **_: calls.__setitem__("kw", calls["kw"] + 1) or 0), \
                patch("library_ops.orchestrate.delete_old_vectorstore_files.run", side_effect=lambda **_: calls.__setitem__("del", calls["del"] + 1) or 0), \
                patch("library_ops.orchestrate._run_smoke_test", side_effect=lambda **_: calls.__setitem__("smoke", calls["smoke"] + 1) or 0), \
                patch("library_ops.orchestrate.archive_version_folder", side_effect=lambda *_, **__: calls.__setitem__("archive", calls["archive"] + 1) or (lib / "archives" / "x.zip")):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = orchestrate.run(
                        manifest=None,
                        apply=False,
                        mode="copy",
                        debug=False,
                        dry_run=True,
                        only_empty_keywords=False,
                        only_empty_vectorstores=False,
                        vs_name_prefix="",
                        upload_missing=False,
                        file_batch_size=100,
                        old_manifest=None,
                        skip_vs="",
                        delete_old_vectors=False,
                        archive_previous=True,
                        keep_old_folder=False,
                        smoke_test=True,
                        smoke_test_query="¿qué es el SIAR?",
                    )

            log = out.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("PLAN: rebuild_library_versioned", log)
            self.assertIn("PLAN: delete_old_vectorstore_files", log)
            self.assertIn("STEP SKIPPED: fill_category_vectorstores", log)
            self.assertIn("STEP SKIPPED: smoke_test_query", log)
            self.assertIn("STEP SKIPPED: archive_previous_version", log)
            self.assertEqual(calls, {"vs": 0, "kw": 0, "del": 0, "smoke": 0, "archive": 0})

    def test_non_dry_run_skips_delete_by_default_and_runs_smoke_and_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            old_manifest = self._mk_version(
                lib,
                "v1",
                {
                    "categories": {"cat": {"vector_store_id": "vs_old", "keywords": []}},
                    "docs": {"d1": {"category": "cat", "vector_store_id": "vs_old", "openai_file_id": "file_1"}},
                },
            )
            new_manifest = self._mk_version(
                lib,
                "v2",
                {
                    "categories": {"cat": {"vector_store_id": "", "keywords": []}},
                    "docs": {"d1": {"category": "cat", "vector_store_id": "", "openai_file_id": ""}},
                },
            )

            def fake_fill_vectorstores(**kwargs):
                data = json.loads(new_manifest.read_text(encoding="utf-8"))
                data["categories"]["cat"]["vector_store_id"] = "vs_new"
                data["docs"]["d1"]["vector_store_id"] = "vs_new"
                new_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return 0

            with patch("library_ops.orchestrate.get_biblioteca_root", return_value=lib), \
                patch("library_ops.orchestrate.rebuild_versioned.run", return_value=0), \
                patch("library_ops.orchestrate.fill_vectorstores.run", side_effect=fake_fill_vectorstores) as vs_mock, \
                patch("library_ops.orchestrate.fill_keywords.run", return_value=0), \
                patch("library_ops.orchestrate.delete_old_vectorstore_files.run", return_value=0) as del_mock, \
                patch("library_ops.orchestrate._run_smoke_test", return_value=0) as smoke_mock, \
                patch("library_ops.orchestrate.archive_version_folder", return_value=lib / "archives" / "v1.zip") as archive_mock:
                rc = orchestrate.run(
                    manifest=None,
                    apply=True,
                    mode="copy",
                    debug=False,
                    dry_run=False,
                    only_empty_keywords=False,
                    only_empty_vectorstores=False,
                    vs_name_prefix="",
                    upload_missing=False,
                    file_batch_size=100,
                    old_manifest=None,
                    skip_vs="",
                    delete_old_vectors=False,
                    archive_previous=True,
                    keep_old_folder=False,
                    smoke_test=True,
                    smoke_test_query="¿qué es el SIAR?",
                )

            self.assertEqual(rc, 0)
            self.assertEqual(vs_mock.call_count, 1)
            self.assertEqual(del_mock.call_count, 0)
            self.assertEqual(smoke_mock.call_count, 1)
            self.assertEqual(archive_mock.call_count, 1)
            self.assertEqual(archive_mock.call_args.kwargs["keep_old_folder"], False)
            self.assertEqual(old_manifest, lib / "v1" / "_state" / "library.json")

            updated = json.loads(new_manifest.read_text(encoding="utf-8"))
            self.assertEqual(updated["categories"]["cat"]["vector_store_id"], "vs_new")
            self.assertEqual(updated["docs"]["d1"]["vector_store_id"], "vs_new")

    def test_delete_old_vectors_runs_only_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            old_manifest = self._mk_version(lib, "v1", {"categories": {}, "docs": {}})
            self._mk_version(lib, "v2", {"categories": {}, "docs": {}})

            with patch("library_ops.orchestrate.get_biblioteca_root", return_value=lib), \
                patch("library_ops.orchestrate.rebuild_versioned.run", return_value=0), \
                patch("library_ops.orchestrate.fill_vectorstores.run", return_value=0), \
                patch("library_ops.orchestrate.fill_keywords.run", return_value=0), \
                patch("library_ops.orchestrate.delete_old_vectorstore_files.run", return_value=0) as del_mock, \
                patch("library_ops.orchestrate._run_smoke_test", return_value=0), \
                patch("library_ops.orchestrate.archive_version_folder", return_value=lib / "archives" / "v1.zip"):
                rc = orchestrate.run(
                    manifest=None,
                    apply=True,
                    mode="copy",
                    debug=False,
                    dry_run=False,
                    only_empty_keywords=False,
                    only_empty_vectorstores=False,
                    vs_name_prefix="",
                    upload_missing=False,
                    file_batch_size=100,
                    old_manifest=None,
                    skip_vs="",
                    delete_old_vectors=True,
                    archive_previous=False,
                    keep_old_folder=False,
                    smoke_test=False,
                    smoke_test_query="¿qué es el SIAR?",
                )

            self.assertEqual(rc, 0)
            self.assertEqual(del_mock.call_count, 1)
            self.assertEqual(del_mock.call_args.kwargs["manifest"], old_manifest)


if __name__ == "__main__":
    unittest.main()
