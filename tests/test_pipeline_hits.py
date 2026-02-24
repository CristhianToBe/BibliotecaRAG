from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Avoid OpenAI client/prompt init errors during imports in this isolated unit test.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
_prompt_tmp = tempfile.mkdtemp(prefix="prompts_")
os.environ.setdefault("PROMPTS_DIR", _prompt_tmp)
for name in ("confirm_system", "query_arbiter_system", "query_pick_system", "query_refiner_a1_system", "query_refiner_a2_system", "query_refiner_a3_system", "query_writer_system"):
    Path(_prompt_tmp, f"{name}.txt").write_text("{}", encoding="utf-8")

from worklib.query import pipeline
from worklib.store import Category, Doc, Manifest, save_manifest


class PipelineHitsRegressionTest(unittest.TestCase):
    def test_hits_forwarded_to_arbiter_when_category_has_vs_and_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "manifest.json"
            manifest = Manifest(
                categories={
                    "doctrina": Category(name="doctrina", vector_store_id="vs_123", folder="docs"),
                },
                docs={
                    "d1": Doc(
                        doc_id="d1",
                        filename="siar.pdf",
                        abs_path=str(Path(tmp) / "siar.pdf"),
                        category="doctrina",
                        openai_file_id="file_1",
                        vector_store_id="vs_123",
                    )
                },
            )
            save_manifest(mp, manifest)

            captured = {}

            def fake_retrieve(vs_ids, query, **kwargs):
                if vs_ids == ["vs_123"] and "SIAR" in query.upper():
                    return [{"file_id": "file_1", "filename": "siar.pdf", "score": 0.9, "text": "SIAR definición"}]
                return []

            def fake_arbitrate(question, refiners, hits, **kwargs):
                captured["hits_len"] = len(hits)
                return {"best_query": question, "reason": "ok"}

            with patch("worklib.query.pipeline.pick_categories", return_value={"selected": ["doctrina"]}), \
                patch("worklib.query.pipeline.confirm_loop", side_effect=lambda question, **kwargs: (question, ["doctrina"], {"selected": ["doctrina"]})), \
                patch("worklib.query.pipeline.refine_all", return_value=[]), \
                patch("worklib.query.pipeline.retrieve_via_tool", side_effect=fake_retrieve), \
                patch("worklib.query.pipeline.arbitrate", side_effect=fake_arbitrate), \
                patch("worklib.query.pipeline.write_answer", return_value="ok"):
                result = pipeline.pro_query_with_meta("¿qué es el SIAR?", manifest_path=str(mp), confirm=True, debug=False)

            self.assertEqual(result["answer"], "ok")
            self.assertGreater(captured.get("hits_len", 0), 0)


if __name__ == "__main__":
    unittest.main()
