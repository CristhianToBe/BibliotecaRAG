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
            self._mk_version(lib, "v0001", {"categories": {}, "docs": {}})

            calls = {"vs": 0, "kw": 0, "del": 0}

            def fake_default_config():
                return type("Cfg", (), {"library_dir": str(lib)})()

            with patch("library_ops.orchestrate.rebuild_versioned.run", return_value=0), \
                patch("library_ops.orchestrate.fill_vectorstores.run", side_effect=lambda **_: calls.__setitem__("vs", calls["vs"] + 1) or 0), \
                patch("library_ops.orchestrate.fill_keywords.run", side_effect=lambda **_: calls.__setitem__("kw", calls["kw"] + 1) or 0), \
                patch("library_ops.orchestrate.delete_old_vectorstore_files.run", side_effect=lambda **_: calls.__setitem__("del", calls["del"] + 1) or 0), \
                patch("worklib.config.default_config", side_effect=fake_default_config):
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
                    )

            log = out.getvalue()
            self.assertEqual(rc, 0)
            self.assertIn("PLAN: rebuild_library_versioned", log)
            self.assertIn("PLAN: fill_category_vectorstores", log)
            self.assertIn("PLAN: fill_category_keywords", log)
            self.assertIn("PLAN: delete_old_vectorstore_files", log)
            self.assertIn("STEP SKIPPED: fill_category_vectorstores", log)
            self.assertIn("STEP SKIPPED: delete_old_vectorstore_files", log)
            self.assertEqual(calls, {"vs": 0, "kw": 0, "del": 0})

    def test_non_dry_run_executes_vectorstores_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp)
            old_manifest = self._mk_version(
                lib,
                "v0001",
                {
                    "categories": {"cat": {"vector_store_id": "vs_old", "keywords": []}},
                    "docs": {"d1": {"category": "cat", "vector_store_id": "vs_old", "openai_file_id": "file_1"}},
                },
            )
            new_manifest = self._mk_version(
                lib,
                "v0002",
                {
                    "categories": {"cat": {"vector_store_id": "", "keywords": []}},
                    "docs": {"d1": {"category": "cat", "vector_store_id": "", "openai_file_id": ""}},
                },
            )

            def fake_default_config():
                return type("Cfg", (), {"library_dir": str(lib)})()

            def fake_fill_vectorstores(**kwargs):
                data = json.loads(new_manifest.read_text(encoding="utf-8"))
                data["categories"]["cat"]["vector_store_id"] = "vs_new"
                data["docs"]["d1"]["vector_store_id"] = "vs_new"
                new_manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                return 0

            with patch("library_ops.orchestrate.rebuild_versioned.run", return_value=0), \
                patch("library_ops.orchestrate.fill_vectorstores.run", side_effect=fake_fill_vectorstores) as vs_mock, \
                patch("library_ops.orchestrate.fill_keywords.run", return_value=0), \
                patch("library_ops.orchestrate.delete_old_vectorstore_files.run", return_value=0) as del_mock, \
                patch("worklib.config.default_config", side_effect=fake_default_config):
                out = io.StringIO()
                with redirect_stdout(out):
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
                    )

            self.assertEqual(rc, 0)
            self.assertEqual(vs_mock.call_count, 1)
            self.assertEqual(del_mock.call_count, 1)
            self.assertEqual(del_mock.call_args.kwargs["manifest"], old_manifest)

            updated = json.loads(new_manifest.read_text(encoding="utf-8"))
            self.assertEqual(updated["categories"]["cat"]["vector_store_id"], "vs_new")
            self.assertEqual(updated["docs"]["d1"]["vector_store_id"], "vs_new")


if __name__ == "__main__":
    unittest.main()
