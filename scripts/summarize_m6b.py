#!/usr/bin/env python3
"""Summarize the M6b gate runs (Gate A no-regression + Gate B fixed-fidelity)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
M2 = REPO / "experiments" / "2026-09-02-m2-topkl-calibration"
sys.path.insert(0, str(M2))

from analyze_m2 import macro_from_domain, points_from_logs  # noqa: E402

GI = 1024 ** 3
DOMAINS = ["wiki_test", "wiki_valid", "chinese", "code", "agent_chat"]
ALL_POINTS = points_from_logs("orcarouter")


def macro(tag: str) -> tuple[float | None, float | None, int | None]:
    doms = ALL_POINTS.get(tag, {})
    metrics = {d: m for d, m in doms.items() if "mean_kld" in m}
    if not metrics:
        return None, None, None
    mac, _ = macro_from_domain(metrics)
    return mac["macro_kld"], mac["macro_same_top"], len(metrics)


def size_of(tag: str) -> int | None:
    for line in (M2 / "results" / "artifact-manifest.txt").read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == f"orcarouter-{tag}":
            return int(parts[1])
    return None


def planned_size(prefix: str) -> int | None:
    plan = M2 / f"{prefix}-plan.json"
    if plan.is_file():
        return json.loads(plan.read_text())["predicted_size_bytes"]
    return None


def fmt(mac: tuple, size: int | None) -> str:
    kl, top, n = mac
    if kl is None:
        return "PENDING"
    s = f"{size / GI:.2f}G" if size else "?"
    return f"kld {kl:.4f} top {top:.2f} (n={n}) @ {s}"


def main() -> int:
    rows: list[tuple[str, str, str]] = []

    # ---- Gate A: crossing targets, no regression vs bootstrap-v0 ----
    bootstrap = {
        "Quality": ("FIT-V2-Quality", 0.0523, 17388186848),
        "Balanced": ("FIT-V2-Balanced", 0.0976, 13900299488),
        "Compact": ("FIT-V2-Compact", 0.1746, 12259175648),
        "Mini": ("FIT-V2-Mini", 0.1924, 11116944608),
    }
    print("== Gate A: Fixed Size (crossing targets) — v2b vs bootstrap-v0 ==")
    for tier, (tag, ref_kl, ref_bytes) in bootstrap.items():
        kl, top, n = macro(tag.replace("FIT-V2", "FIT-V2B"))
        size = size_of(tag.replace("FIT-V2", "FIT-V2B"))
        note = ""
        if kl is None:
            note = "identity pass expected (recipe byte-equal to bootstrap)" if tier in ("Compact", "Mini") else ""
        print(f"  {tier:9s} v2b: {fmt((kl, top, n), size):34s} bootstrap: kld {ref_kl:.4f} @ {ref_bytes/GI:.2f}G  {note}")

    # ---- Gate B: fixed fidelity, Size_v2b <= Size_v0.1 ----
    print("\n== Gate B: Fixed Fidelity — v2b at v0.1 sizes vs tier anchors ==")
    v01 = {
        "Balanced": ("FIT-13G (v0.1)", 0.0987, 13956046016, 0.10),
        "Compact": ("FIT-11.5G (v0.1)", 0.1439, 12277279936, 0.15),
        "Mini": ("FIT-A-IQ3XXS-v01 (v0.1)", 0.1955, 11186330848, 0.20),
    }
    v2b_tags = {
        "Balanced": "FIT-V2B-BalancedFF",
        "Compact": "FIT-V2B-CompactFF",
        "Mini": "FIT-V2-Mini",  # identity: recipe at crossing target == bootstrap
    }
    for tier, (label, v01_kl, v01_bytes, anchor) in v01.items():
        tag = v2b_tags[tier]
        kl, top, n = macro(tag)
        size = size_of(tag)
        verdict = "PENDING"
        if kl is not None and size is not None:
            ok = kl <= anchor and size <= v01_bytes
            verdict = "PASS" if ok else "FAIL"
        print(f"  {tier:9s} v2b({tag}): {fmt((kl, top, n), size):34s} | v0.1 {label}: kld {v01_kl:.4f} @ {v01_bytes/GI:.2f}G | anchor {anchor:.2f} -> {verdict}")

    # ---- Compact window-confound isolation ----
    print("\n== Compact confound isolation @ 12,277,279,936 (v0.1 bytes) ==")
    for tag, note in (
        ("FIT-V1B-CompactFF", "v1 stack, IQ3_XS->IQ3_S (clean control)"),
        ("FIT-V2B-CompactFF", "band profile, IQ3_XS->IQ3_S"),
        ("FIT-V2B-CompactFFQ3", "band profile, Q3_K_S->IQ3_S (poison floor)"),
    ):
        kl, top, n = macro(tag)
        size = size_of(tag)
        print(f"  {tag:22s} {fmt((kl, top, n), size):34s} | {note}")
    print("  bootstrap V2-Compact (Q3_K_S floor, 11.42G): kld 0.1746")
    print("  v0.1 FIT-11.5G (IQ3_XS floor, 11.43G):      kld 0.1439")
    return 0


if __name__ == "__main__":
    sys.exit(main())
