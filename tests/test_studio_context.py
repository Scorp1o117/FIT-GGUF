from copy import deepcopy

import pytest

from fit_gguf.studio.context import check_model_context, check_plan_context


def records(tmp_path):
    paths = {key: str(tmp_path / key) for key in ("source", "imatrix", "runtime")}
    analysis = {"source": {"path": paths["source"]}, "imatrix": {"path": paths["imatrix"]},
                "runtime": {"dir": paths["runtime"]},
                "presets": {"lower": {"name": "IQ3_M"}, "upper": {"name": "IQ4_XS"}}}
    return analysis, {"model_context": paths | {"lower": "IQ3_M", "upper": "IQ4_XS"}}


@pytest.mark.parametrize("field", ["source", "imatrix", "runtime", "lower", "upper"])
def test_changed_visible_model_context_refused(tmp_path, field):
    analysis, payload = records(tmp_path)
    check_model_context(payload, analysis)
    payload["model_context"][field] = "changed"
    with pytest.raises(ValueError, match="differ"):
        check_model_context(payload, analysis)


@pytest.mark.parametrize("context", [[], {}, {"source": "bad"}])
def test_invalid_context_refused(tmp_path, context):
    analysis, _ = records(tmp_path)
    with pytest.raises(ValueError, match="Invalid"):
        check_model_context({"model_context": context}, analysis)


def test_record_only_clients_still_supported(tmp_path):
    analysis, _ = records(tmp_path)
    check_model_context({}, analysis)
    check_plan_context({}, {"target_bytes": 10, "policy": "balanced"})


@pytest.mark.parametrize("context", [
    {"target_bytes": 11, "policy": "balanced"}, {"target_bytes": True, "policy": "balanced"},
    {"target_bytes": 10, "policy": "original"}, {"target_bytes": "10", "policy": "balanced"},
])
def test_changed_budget_or_policy_refused(context):
    plan = {"target_bytes": 10, "policy": "balanced"}
    check_plan_context({"plan_context": deepcopy(plan)}, plan)
    with pytest.raises(ValueError):
        check_plan_context({"plan_context": context}, plan)
