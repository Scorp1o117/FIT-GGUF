"""Check that the visible Studio inputs still describe an existing record."""
from pathlib import Path


def check_model_context(payload, analysis):
    context = payload.get("model_context")
    if context is None:  # Older API clients still use record/hash validation.
        return
    if not isinstance(context, dict) or set(context) != {"source", "imatrix", "runtime", "lower", "upper"}:
        raise ValueError("Invalid model context")
    expected = {"source": analysis["source"]["path"], "imatrix": analysis["imatrix"]["path"],
                "runtime": analysis["runtime"]["dir"]}
    for key, path in expected.items():
        value = context[key]
        if not isinstance(value, str) or not value.strip() or Path(value.strip()).expanduser().resolve() != Path(path).resolve():
            raise ValueError(f"Current {key} differs from this analysis. Analyze again or load the matching record.")
    for key in ("lower", "upper"):
        if context[key] != analysis["presets"][key]["name"]:
            raise ValueError("Current presets differ from this analysis. Analyze again.")


def check_plan_context(payload, plan):
    context = payload.get("plan_context")
    if context is None:
        return
    if not isinstance(context, dict) or set(context) != {"target_bytes", "policy"}:
        raise ValueError("Invalid plan context")
    target = context["target_bytes"]
    if isinstance(target, bool) or not isinstance(target, int) or target != plan["target_bytes"]:
        raise ValueError("Current budget differs from this plan. Generate a new plan.")
    if context["policy"] != plan["policy"]:
        raise ValueError("Current allocation policy differs from this plan. Generate a new plan.")
