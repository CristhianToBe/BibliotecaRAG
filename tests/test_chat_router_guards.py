from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module(rel_path: str, name: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    spec = spec_from_file_location(name, path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_turn_refine_routes_to_awaiting_user_reply() -> None:
    rules = _load_module("worklib/query/confirm_rules.py", "confirm_rules_local")
    assert rules.route_confirm_action(action="REFINE", user_reply="") == "AWAITING_USER_REPLY"


def test_second_turn_refine_routes_to_repick() -> None:
    rules = _load_module("worklib/query/confirm_rules.py", "confirm_rules_local2")
    assert rules.route_confirm_action(action="REFINE", user_reply="no, enfócate") == "REPICK"


def test_domain_hint_detected_for_pasivos_inexistentes() -> None:
    hints = _load_module("worklib/query/intent_hints.py", "intent_hints_local")
    assert hints.detect_domain_hint("cual es la sanción por pasivos inexistentes") == "TRIBUTARIO"


def test_domain_hint_protects_dian_terms() -> None:
    hints = _load_module("worklib/query/intent_hints.py", "intent_hints_local2")
    out = hints.apply_domain_hint_constraints(
        domain_hint="TRIBUTARIO",
        must_terms=[],
        avoid_terms=["DIAN", "otra"],
        query_text="pasivos inexistentes",
    )
    assert "DIAN" in out["must_terms"]
    assert "DIAN" not in out["avoid_terms"]


def test_manifest_guard_shape() -> None:
    guards = _load_module("worklib/query/pipeline_guards.py", "pipeline_guards_local")
    out = guards.manifest_not_loaded_response(trace_id="t1", manifest_path="/tmp/m.json", manifest_error="FileNotFound")
    assert out["status"] == "error"
    assert out["error"] == "MANIFEST_NOT_LOADED"
    assert "NO_VALID_CATEGORIES" in out["details"]
