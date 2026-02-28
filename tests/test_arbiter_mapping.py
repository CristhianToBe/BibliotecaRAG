from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_arbiter_utils_module():
    path = Path(__file__).resolve().parents[1] / "worklib" / "query" / "arbiter_utils.py"
    spec = spec_from_file_location("arbiter_utils_local", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derive_winner_from_selected_index() -> None:
    mod = _load_arbiter_utils_module()
    considered = ["A1", "A2", "A3"]
    winner = mod.derive_winner(considered=considered, winner="", selected_indexes=[2])
    assert winner == "A3"


def test_derive_winner_prefers_explicit_name() -> None:
    mod = _load_arbiter_utils_module()
    considered = ["A1", "A2", "A3"]
    winner = mod.derive_winner(considered=considered, winner="A2", selected_indexes=[2])
    assert winner == "A2"
