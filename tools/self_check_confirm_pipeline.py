from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(rel_path: str, name: str):
    path = ROOT / rel_path
    spec = spec_from_file_location(name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    rules = _load("worklib/query/confirm_rules.py", "confirm_rules_local")
    hints = _load("worklib/query/intent_hints.py", "intent_hints_local")
    guards = _load("worklib/query/pipeline_guards.py", "pipeline_guards_local")

    out = [
        {
            "case": "manifest_empty",
            "result": guards.manifest_not_loaded_response(
                trace_id="t1",
                manifest_path="/missing/manifest.json",
                manifest_error="FileNotFound",
            ),
        },
        {
            "case": "first_turn_confirm_refine_no_autorepick",
            "result": {
                "user_reply": "",
                "route": rules.route_confirm_action(action="REFINE", user_reply=""),
            },
        },
        {
            "case": "domain_hint_pasivos_inexistentes",
            "result": {
                "question": "cual es la sanción por pasivos inexistentes",
                "domain_hint": hints.detect_domain_hint("cual es la sanción por pasivos inexistentes"),
            },
        },
    ]
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
