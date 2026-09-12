#!/usr/bin/env python3
"""Chart APEX vs FIT vs native llama.cpp presets for Nex-N2.5-mini-abliterated.

All three families are measured with the identical eval-v1 five-domain KL
protocol (llama-perplexity --kl-divergence against the same bf16-<domain>.kld
references), so the points are directly comparable.

    python scripts/plot_apex_vs_fit.py \
        --eval experiments/2026-09-11-nex25-mini-4tier/apex-vs-fit-eval.json \
        --calibration experiments/2026-09-11-nex25-mini-4tier/calibration/curve-points.jsonl \
        --out experiments/2026-09-11-nex25-mini-4tier/apex-vs-fit-vs-native.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIER_GATES = [0.05, 0.10, 0.15, 0.20]
GIB = 2**30

# Curve points that are real llama.cpp presets (probe-* entries are FIT gap probes).
NATIVE_SKIP = ("probe-",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers-json", default=None, help="optional JSON of FIT tier results")
    args = ap.parse_args()

    ev = json.loads(Path(args.eval).read_text())
    apex, fit = [], []
    for a in ev["artifacts"]:
        point = (a["size_bytes"] / GIB, a["macro_kl"], a["name"])
        (apex if "APEX" in a["name"] else fit).append(point)

    native = []
    for line in Path(args.calibration).read_text().splitlines():
        p = json.loads(line)
        if any(p["point_id"].startswith(s) for s in NATIVE_SKIP):
            continue
        native.append((p["size_bytes"] / GIB, p["macro_kl"], p["point_id"]))

    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)

    # tier gates
    for g in TIER_GATES:
        ax.axhline(g, color="#cccccc", lw=1, ls=":", zorder=1)
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] else 0, g, "", va="center")

    native.sort()
    ax.plot([p[0] for p in native], [p[1] for p in native], "o-",
            color="#888888", ms=6, lw=1.4, label="native llama.cpp presets", zorder=3)
    for x, y, name in native:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color="#666666")

    fit.sort()
    ax.plot([p[0] for p in fit], [p[1] for p in fit], "s-",
            color="#1f77b4", ms=9, lw=2, label="FIT (KL-anchored tiers)", zorder=4)
    for x, y, name in fit:
        tier = name.split("FIT-")[1].split("-")[0]
        ax.annotate(tier, (x, y), textcoords="offset points", xytext=(7, -13),
                    fontsize=9, color="#1f77b4", fontweight="bold")

    apex.sort()
    ax.plot([p[0] for p in apex], [p[1] for p in apex], "^-",
            color="#d62728", ms=10, lw=2, label="APEX-I (LocalAI MoE profiles)", zorder=5)
    for x, y, name in apex:
        tier = name.split("APEX-")[1].replace(".gguf", "")
        ax.annotate(tier, (x, y), textcoords="offset points", xytext=(7, 7),
                    fontsize=9, color="#d62728", fontweight="bold")

    for g in TIER_GATES:
        ax.axhline(g, color="#bbbbbb", lw=1, ls="--", zorder=1)
        ax.annotate(f"tier gate {g:.2f}", (ax.get_xlim()[1], g), xytext=(-4, 3),
                    textcoords="offset points", ha="right", fontsize=7, color="#999999")

    # Headroom so the lowest point (APEX I-Balanced) is never clipped, and the
    # small-size labels (FIT MINI / APEX Mini) do not collide.
    ys = [p[1] for p in native + fit + apex]
    ax.set_ylim(0, max(ys) * 1.08)
    ax.set_xlim(min(p[0] for p in native + fit + apex) - 0.6,
                max(p[0] for p in native + fit + apex) + 1.6)

    ax.set_xlabel("artifact size (GiB)")
    ax.set_ylabel("macro KL vs BF16 (lower is better)")
    ax.set_title("Nex-N2.5-mini-abliterated — APEX vs FIT vs native\n"
                 "identical five-domain eval-v1 KL protocol, llama.cpp b10666 (ROCm)",
                 fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")

    rows = sorted(native + fit + apex, key=lambda r: r[0])
    print(f"\n  {'family':8s} {'size GiB':>9s} {'macro KL':>9s}  name")
    for x, y, name in rows:
        fam = "APEX" if "APEX" in name else ("FIT" if "FIT-" in name else "native")
        print(f"  {fam:8s} {x:9.2f} {y:9.4f}  {name}")


if __name__ == "__main__":
    main()
