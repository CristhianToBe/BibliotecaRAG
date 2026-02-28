from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_utils_module():
    path = Path(__file__).resolve().parents[1] / "worklib" / "query" / "arbiter_utils.py"
    spec = spec_from_file_location("arbiter_utils_local", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_variant_name_is_signal_based() -> None:
    mod = _load_utils_module()
    considered = ["A1", "A2", "A3"]
    signal_summary = {
        "A1": {"must_terms_coverage": 0.6, "unique_docs": 4, "hits_count": 9},
        "A2": {"must_terms_coverage": 0.8, "unique_docs": 2, "hits_count": 20},
        "A3": {"must_terms_coverage": 0.8, "unique_docs": 5, "hits_count": 1},
    }
    assert mod.choose_variant_name(signal_summary, considered) == "A3"


def test_validate_evidence_indexes_fallbacks_when_invalid() -> None:
    mod = _load_utils_module()
    selected, also, used_fallback = mod.validate_evidence_indexes(
        selected_indexes=[100, -1],
        also_indexes=[99],
        evidence_len=4,
        fallback_order=[2, 1, 0, 3],
    )
    assert used_fallback is True
    assert selected == [2, 1, 0]
    assert also == []
