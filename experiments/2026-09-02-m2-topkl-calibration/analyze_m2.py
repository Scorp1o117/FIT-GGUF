#!/usr/bin/env python3
"""M2 Top|KL candidate calibration analysis.

Implements the preregistered rules of
experiments/2026-09-02-m2-topkl-calibration/PREREGISTRATION.md §7:
  Layer 1  per-model Top@KL crossing on the Pareto frontier (linear interp)
  Layer 2  per-model pre-registered window health P5/P10 (no pooled percentiles)
  Layer 3  equal-model aggregate
Mechanical verdicts are preliminary evidence; the planner judges.

Stdlib only. Run: python3 analyze_m2.py
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path("/run/media/s117/OS/FIT-GGUF")
M2 = ROOT / "experiments/2026-09-02-m2-topkl-calibration"
sys.path.insert(0, str(ROOT / "src"))

from fit_gguf.eval.results import parse_llama_kl_log  # noqa: E402

DOMAINS = ["wiki_test", "wiki_valid", "chinese", "code", "agent_chat"]
CANDIDATES = {0.05: 93.0, 0.10: 91.0, 0.15: 88.0, 0.20: 85.0}
WINDOWS = {t: (round(0.85 * t, 6), round(1.15 * t, 6)) for t in CANDIDATES}
LOGDIR = M2 / "logs"
DOMAIN_RE = re.compile(r"-(wiki_test|wiki_valid|chinese|code|agent_chat)\.log$")


def load_ling():
    """Ling-3.0-tiny (bailingmoe) — third DEV family, all fresh points."""
    points = []
    manifest = {}
    mf = M2 / "results-ling" / "artifact-manifest.txt"
    if mf.exists():
        for line in mf.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                manifest[parts[0]] = {"bytes": int(parts[1]), "sha256": parts[2]}
    for point, doms in points_from_logs("ling", M2 / "logs-ling").items():
        mac, missing = macro_from_domain(doms)
        kind = "fit" if ("APEX" in point or point.startswith("FIT-")) else "preset"
        entry = {"name": point, "kind": kind, "bytes": manifest.get("ling-" + point, {}).get("bytes"),
                 "source": "m2-ling-new", "per_domain": doms, "flags": []}
        if mac is None:
            entry.update({"incomplete_domains": missing, "macro_kld": None, "macro_same_top": None,
                          "flags": entry["flags"] + ["incomplete"]})
        else:
            entry.update(mac)
        points.append(entry)
    return points


def load_gemma():
    """gemma-4-E4B (dense text tower) — third dense DEV family, all fresh points.

    Prereg G-A1: the three low-bit preset points (IQ2_XXS/IQ2_XS/IQ3_XXS,
    plus IQ2_M/IQ3_XS whose recipes contain requires-imatrix dst types) carry
    18 attn_k tensors promoted to Q4_K (b10666 imatrix collection gap for
    gemma4); all other points are pure presets.
    """
    points = []
    manifest = {}
    mf = M2 / "results-gemma" / "artifact-manifest.txt"
    if mf.exists():
        for line in mf.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                manifest[parts[0]] = {"bytes": int(parts[1]), "sha256": parts[2]}
    for point, doms in points_from_logs("gemma", M2 / "logs-gemma").items():
        mac, missing = macro_from_domain(doms)
        kind = "fit" if point.startswith("FIT-") else "preset"
        entry = {"name": point, "kind": kind, "bytes": manifest.get("gemma-" + point, {}).get("bytes"),
                 "source": "m2-gemma-new", "per_domain": doms, "flags": []}
        if mac is None:
            entry.update({"incomplete_domains": missing, "macro_kld": None, "macro_same_top": None,
                          "flags": entry["flags"] + ["incomplete"]})
        else:
            entry.update(mac)
        points.append(entry)
    return points


def macro_from_domain(domain_metrics):
    missing = [d for d in DOMAINS if d not in domain_metrics]
    if missing:
        return None, missing
    kl = sum(domain_metrics[d]["mean_kld"] for d in DOMAINS) / len(DOMAINS)
    top = sum(domain_metrics[d]["same_top_pct"] for d in DOMAINS) / len(DOMAINS)
    rms = [domain_metrics[d].get("rms_delta_p_pct") for d in DOMAINS]
    rms = sum(r for r in rms if r is not None) / len([r for r in rms if r is not None]) if any(r is not None for r in rms) else None
    return {"macro_kld": kl, "macro_same_top": top, "rms_delta_p_pct": rms}, []


def points_from_logs(model, logdir=None):
    """Group eval-<model>-<point>-<domain>.log files into curve points.

    Enumerates via os.listdir: ntfs3 readdir/glob intermittently drops
    freshly created entries from a large directory (observed 2026-09-03 with
    the FIT-V2 logs); listdir has been reliable.
    """
    import os

    base = Path(logdir) if logdir else LOGDIR
    prefix = f"eval-{model}-"
    by_point = {}
    for name in sorted(os.listdir(base)):
        if not name.startswith(prefix) or not name.endswith(".log"):
            continue
        m = DOMAIN_RE.search(name)
        if not m:
            continue
        domain = m.group(1)
        point = name[len(prefix) : -len(m.group(0))].rstrip("-")
        try:
            metrics = parse_llama_kl_log((base / name).read_text(errors="replace"))
        except Exception as e:  # noqa: BLE001 - record as-is
            by_point.setdefault(point, {})[domain] = {"error": str(e)}
            continue
        by_point.setdefault(point, {})[domain] = metrics
    return by_point


def load_manifest():
    manifest = {}
    mf = M2 / "results" / "artifact-manifest.txt"
    if not mf.exists():
        return manifest
    for line in mf.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            manifest[parts[0]] = {"bytes": int(parts[1]), "sha256": parts[2]}
        elif len(parts) == 2 and parts[1] == "MISSING":
            manifest[parts[0]] = {"bytes": None, "sha256": None}
    return manifest


def load_orcarouter():
    points = []
    p4 = json.loads((ROOT / "experiments/2026-08-29-p4-release-batch/results/p4-results.json").read_text())
    for name, a in p4["artifacts"].items():
        doms = {
            d: {"mean_kld": v["mean_kld"], "same_top_pct": v["same_top"], "rms_delta_p_pct": None}
            for d, v in a["domains"].items()
        }
        mac, _ = macro_from_domain(doms)
        flags = []
        if abs(mac["macro_kld"] - a["macro_kld"]) > 5e-7:
            flags.append("published_macro_kld_mismatch")
        if abs(mac["macro_same_top"] - a["macro_same_top"]) > 0.005:
            flags.append("published_macro_top_mismatch")
        points.append({
            "name": name,
            "kind": "fit" if a.get("kind") == "fit" else "preset",
            "bytes": a["actual_bytes"],
            "source": "p4-published (reuse, corpus-hash verified)",
            "per_domain": doms,
            **mac,
            "flags": flags,
        })
    manifest = load_manifest()
    for point, doms in points_from_logs("orcarouter").items():
        mac, missing = macro_from_domain(doms)
        if mac is None:
            points.append({"name": point, "kind": "preset", "bytes": manifest.get("orcarouter-" + point, {}).get("bytes"),
                           "source": "m2-new", "per_domain": doms, "incomplete_domains": missing,
                           "macro_kld": None, "macro_same_top": None, "flags": ["incomplete"]})
            continue
        kind = "fit" if "FIT" in point else "preset"
        # main manifest keys carry the model prefix for every m2-new point
        # (orcarouter-FIT-14G, orcarouter-M2A, ...); fall back to the bare
        # name for historical records.
        entry_bytes = manifest.get("orcarouter-" + point, {}).get("bytes")
        if entry_bytes is None:
            entry_bytes = manifest.get(point, {}).get("bytes")
        points.append({"name": point, "kind": kind, "bytes": entry_bytes,
                       "source": "m2-new", "per_domain": doms, **mac, "flags": []})
    return points


def load_granite():
    points = []
    manifest = load_manifest()
    fit_expect = {"O-FIT25": 4271391104, "B-FIT25": 4271931776, "O-FIT50": 4454351232,
                  "B-FIT50": 4454564224, "O-FIT75": 4637311360, "B-FIT75": 4637311360}
    for point, doms in points_from_logs("granite").items():
        mac, missing = macro_from_domain(doms)
        kind = "fit" if "FIT" in point else "preset"
        entry = {"name": point, "kind": kind, "bytes": manifest.get("granite-" + point, {}).get("bytes"),
                 "source": "m2-new", "per_domain": doms, "flags": []}
        if kind == "fit":
            entry["expected_bytes"] = fit_expect.get(point)
            if entry["bytes"] is not None and entry["expected_bytes"] is not None and entry["bytes"] != entry["expected_bytes"]:
                entry["flags"].append("size_gate_mismatch")
        if mac is None:
            entry.update({"incomplete_domains": missing, "macro_kld": None, "macro_same_top": None,
                          "flags": entry["flags"] + ["incomplete"]})
        else:
            entry.update(mac)
        points.append(entry)
    return points


def top_at_target(points, target):
    """Amendment-0 rule: nearest-straddle interpolation on the raw curve."""
    valid = sorted((p for p in points if p.get("macro_kld") is not None),
                   key=lambda p: p["macro_kld"])
    if not valid:
        return None, "empty curve"
    below = [p for p in valid if p["macro_kld"] <= target]
    above = [p for p in valid if p["macro_kld"] > target]
    if not below:
        return None, "no point below target"
    if not above:
        return None, "no point above target"
    a, b = below[-1], above[0]
    if a["macro_kld"] == target:
        return a["macro_same_top"], f"anchor {a['name']}"
    frac = (target - a["macro_kld"]) / (b["macro_kld"] - a["macro_kld"])
    val = a["macro_same_top"] + (b["macro_same_top"] - a["macro_same_top"]) * frac
    return val, f"interp {a['name']} -> {b['name']}"


def dominated_names(points):
    """Curve-quality note: points beaten in both coords by another point."""
    return [p["name"] for p in points
            if any(q["macro_kld"] < p["macro_kld"] and q["macro_same_top"] > p["macro_same_top"]
                   for q in points)]


def pct(vals, q):
    vs = sorted(vals)
    n = len(vs)
    if n == 1:
        return vs[0]
    h = (n - 1) * q / 100.0
    lo, hi = math.floor(h), math.ceil(h)
    return vs[lo] + (vs[hi] - vs[lo]) * (h - lo)


def window_health(points, lo, hi):
    inside = [p for p in points if lo <= p["macro_kld"] <= hi]
    if len(inside) < 2:
        return {"window": [lo, hi], "n": len(inside),
                "points": [{"name": p["name"], "kind": p["kind"], "macro_kld": round(p["macro_kld"], 7),
                            "macro_same_top": round(p["macro_same_top"], 4)} for p in inside],
                "p5": None, "p10": None, "min": None, "max": None, "fit_only_n": 0}
    tops = [p["macro_same_top"] for p in inside]
    fit_only = [p for p in inside if p["kind"] == "fit"]
    return {"window": [lo, hi], "n": len(inside),
            "kinds": {"fit": sum(1 for p in inside if p["kind"] == "fit"),
                      "preset": sum(1 for p in inside if p["kind"] == "preset")},
            "p5": round(pct(tops, 5), 4), "p10": round(pct(tops, 10), 4),
            "min": round(min(tops), 4), "max": round(max(tops), 4),
            "fit_only_n": len(fit_only),
            "fit_only": ({"p5": round(pct([p["macro_same_top"] for p in fit_only], 5), 4),
                          "p10": round(pct([p["macro_same_top"] for p in fit_only], 10), 4)}
                         if len(fit_only) >= 2 else None),
            "points": [{"name": p["name"], "kind": p["kind"], "macro_kld": round(p["macro_kld"], 7),
                        "macro_same_top": round(p["macro_same_top"], 4)} for p in inside]}


def verdict(top_at, p10, candidate):
    if top_at is None:
        return "indeterminate (crossing not bracketed)"
    if p10 is None:
        return "indeterminate (window n<2)"
    if top_at >= candidate and p10 >= candidate:
        return "supported"
    if top_at >= candidate:
        return "at-risk"
    return "violated"


def analyze_model(name, points):
    valid = [p for p in points if p.get("macro_kld") is not None]
    layer1, layer2, verdicts = {}, {}, {}
    for t, cand in CANDIDATES.items():
        key = f"{t:.2f}"
        tv, how = top_at_target(valid, t)
        layer1[key] = {"top_at_target": round(tv, 4) if tv is not None else None,
                       "candidate": cand, "evidence": how}
        lo, hi = WINDOWS[t]
        layer2[key] = window_health(valid, lo, hi)
        verdicts[key] = {"candidate": cand, "verdict": verdict(tv, layer2[key].get("p10"), cand),
                         "delta_top_interp_pp": round(tv - cand, 4) if tv is not None else None,
                         "delta_p10_pp": (round(layer2[key]["p10"] - cand, 4)
                                          if layer2[key].get("p10") is not None else None)}
    return {
        "model": name,
        "n_points_total": len(points),
        "n_points_valid": len(valid),
        "n_points_flagged": sum(1 for p in points if p.get("flags")),
        "dominated_curve_points": dominated_names(valid),
        "layer1_top_at_kl_crossing": layer1,
        "layer2_window_health": layer2,
        "mechanical_verdicts": verdicts,
        "points": points,
    }


def aggregate(models):
    out = {}
    for t, cand in CANDIDATES.items():
        key = f"{t:.2f}"
        tops = [m["layer1_top_at_kl_crossing"][key]["top_at_target"] for m in models]
        p5s = [m["layer2_window_health"][key].get("p5") for m in models]
        p10s = [m["layer2_window_health"][key].get("p10") for m in models]
        def mean(vals):
            vs = [v for v in vals if v is not None]
            return round(sum(vs) / len(vs), 4) if vs else None
        out[key] = {"candidate": cand,
                    "equal_model_mean_top_at_target": mean(tops),
                    "equal_model_mean_window_p5": mean(p5s),
                    "equal_model_mean_window_p10": mean(p10s),
                    "n_models_contributing_top": sum(1 for v in tops if v is not None),
                    "n_models_contributing_p10": sum(1 for v in p10s if v is not None)}
    return out


def main():
    models = [analyze_model("orcarouter", load_orcarouter()),
              analyze_model("granite", load_granite()),
              analyze_model("ling", load_ling()),
              analyze_model("gemma", load_gemma())]
    agg = aggregate(models)
    result = {
        "schema": "fit.candidate_calibration.v1",
        "status": "CANDIDATE — not a statistical freeze",
        "preregistration": "PREREGISTRATION.md",
        "candidates": {f"{t:.2f}": c for t, c in CANDIDATES.items()},
        "windows": {f"{t:.2f}": list(WINDOWS[t]) for t in WINDOWS},
        "forbidden": "pooled cross-model percentiles (prereg §7)",
        "models": models,
        "layer3_equal_model_aggregate": agg,
    }
    out_json = M2 / "results" / "candidate-calibration-v1.json"
    out_json.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {out_json}")

    lines = ["# M2 candidate-calibration-v1 — per-model tables", "",
             "Verdicts are mechanical prereg §7 evidence; final judgment: planner.", ""]
    for m in models:
        lines.append(f"## {m['model']}  (points {m['n_points_valid']}/{m['n_points_total']} valid, "
                     f"{m['n_points_flagged']} flagged)")
        lines.append("")
        lines.append("| target | window | n(win) | Top@t interp | P5 | P10 | candidate | verdict |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for t in CANDIDATES:
            key = f"{t:.2f}"
            l1, l2, vd = m["layer1_top_at_kl_crossing"][key], m["layer2_window_health"][key], m["mechanical_verdicts"][key]
            fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—"  # noqa: E731
            lines.append(f"| {key} | [{l2['window'][0]:.4f}, {l2['window'][1]:.4f}] | {l2['n']} | "
                         f"{fmt(l1['top_at_target'])} | {fmt(l2.get('p5'))} | {fmt(l2.get('p10'))} | "
                         f"{vd['candidate']:.0f} | {vd['verdict']} |")
        lines.append("")
    lines.append("## Layer 3 — equal-model aggregate")
    lines.append("")
    lines.append("| target | candidate | mean Top@t | mean P5 | mean P10 | n models (top/p10) |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for key, v in agg.items():
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"  # noqa: E731
        lines.append(f"| {key} | {v['candidate']:.0f} | {fmt(v['equal_model_mean_top_at_target'])} | "
                     f"{fmt(v['equal_model_mean_window_p5'])} | {fmt(v['equal_model_mean_window_p10'])} | "
                     f"{v['n_models_contributing_top']}/{v['n_models_contributing_p10']} |")
    lines.append("")
    out_md = M2 / "results" / "candidate-calibration-v1.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
