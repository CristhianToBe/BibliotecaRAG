from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_confirm_rules_module():
    path = Path(__file__).resolve().parents[1] / "worklib" / "query" / "confirm_rules.py"
    spec = spec_from_file_location("confirm_rules_local", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_first_turn_always_refine() -> None:
    mod = _load_confirm_rules_module()
    out = mod.enforce_confirm_schema({}, user_reply="")
    assert out["action"] == "REFINE"


def test_short_affirmative_pass() -> None:
    mod = _load_confirm_rules_module()
    out = mod.enforce_confirm_schema({}, user_reply="si")
    assert out["action"] == "PASS"


def test_refine_reply_has_selector_instruction() -> None:
    mod = _load_confirm_rules_module()
    out = mod.enforce_confirm_schema({}, user_reply="no, enfócate en tributario")
    assert out["action"] == "REFINE"
    assert str(out["selector_instruction"]).strip() != ""


def test_rephrase_reply_has_single_question_text() -> None:
    mod = _load_confirm_rules_module()
    out = mod.enforce_confirm_schema({}, user_reply="cambia el tema a retención")
    assert out["action"] == "REPHRASE"
    assert str(out["rephrased_question"]).strip() != ""
