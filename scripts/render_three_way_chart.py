#!/usr/bin/env python3
"""Three families, one protocol: native presets vs FIT tiers vs APEX tiers.

The point of this chart is that the comparison is honest. All three families are
measured by the same `eval_artifact` — the same five fixed 64 KiB slices, the same
aligned BF16 references, the same llama.cpp build — so a point's position says
something about its recipe and not about how it was measured. The APEX recipes
are run with **this model's own imatrix**, not APEX's diverse "I-" imatrix, which
isolates the recipe from the calibration corpus; the APEX I-variants would need
that corpus and are a different experiment.

Style comes from `render_release_charts.py` by import, not by copy: the palette,
the frame, the header and the footer are the house ones, so a change there moves
this chart too instead of letting the two drift.

USAGE
    python scripts/render_three_way_chart.py \\
        --calibration experiments/<run>/calibration \\
        --fit         /path/to/<model>-FIT/emit-report.json \\
        --apex        /path/to/<model>-APEX/apex-report.json \\
        --model       occamy-1.0-abliterated \\
        --out-dir     docs/assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fit_gguf.fidelity import TIER_DISPLAY  # noqa: E402

# The house style, imported so this chart cannot drift away from it.
from render_release_charts import (  # noqa: E402
    BG,
    BLUE,
    GRID,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    configure,
    footer,
    frame,
    header,
)

# A third accent, in the same family as ORANGE/BLUE and legible against PANEL.
GREEN = "#3FB950"

GIB = 2**30

# Per-tier label placement, in points, with the anchor side. The low-size end
# packs all three families into ~2 GiB, so a uniform above/below rule overlaps
# there; these were placed by looking at the rendered chart. Anything not listed
# keeps the default.
NATIVE_LABEL_OFFSETS = {
    # The FIT reference point lands almost on top of this preset's marker, and
    # the preset label is the one that can move.
    "Q5_K_M": (0, 16, "center"),
}
FIT_LABEL_OFFSETS = {
    "mini": (-40, -8, "right"),
    "compact": (-30, -30, "right"),
    "balanced": (-16, -34, "right"),
    "reference": (20, 10, "left"),
}
APEX_LABEL_OFFSETS = {
    "mini": (20, -20, "left"),
    "compact": (-6, -34, "right"),
    "balanced": (-16, -28, "right"),
}


def load_native(bundle: Path) -> list[dict]:
    """The standard ladder, as measured by the calibration."""
    contract = json.loads(
        (REPO / "src/fit_gguf/contracts/fidelity-calibration-v1.json").read_text()
    )
    presets = set(contract["ladder_standard_presets"])
    out = []
    for line in (bundle / "curve-points.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        point = json.loads(line)
        if point["point_id"] in presets:
            out.append({
                "name": point["point_id"],
                "gib": point["size_bytes"] / GIB,
                "kl": point["macro_kl"],
                "top": point["same_top"],
            })
    return sorted(out, key=lambda row: row["gib"])


def load_fit(path: Path) -> list[dict]:
    """The shipped FIT tiers, measured on the bytes that ship."""
    payload = json.loads(path.read_text())
    out = []
    for row in payload["tiers"]:
        shipped = row.get("shipped")
        out.append({
            # The tier key stays lowercase for offsets and lookups; what a reader
            # sees is the product name, which is capitalized everywhere else.
            "key": row["tier"],
            "name": TIER_DISPLAY.get(row["tier"], row["tier"]),
            "gib": (shipped["size_bytes"] if shipped else row["size_bytes"]) / GIB,
            "kl": (shipped["macro_kl"] if shipped else row["macro_kl"]),
            "top": (shipped["same_top"] if shipped else row["same_top"]),
            "anchor": row.get("anchor"),
        })
    return sorted(out, key=lambda row: row["gib"])


def load_apex(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return sorted(
        (
            {
                "name": row["tier"],
                "gib": row["size_bytes"] / GIB,
                "kl": row["macro_kl"],
                "top": row["same_top"],
                "base": row.get("base_type"),
            }
            for row in payload["tiers"]
            if "macro_kl" in row
        ),
        key=lambda row: row["gib"],
    )


def render(native, fit, apex, model: str, out_dir: Path, lang: str, gates: list[float]) -> Path:
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.18)
    header(
        fig,
        "三大家族，同一套协议" if zh else "THREE FAMILIES, ONE PROTOCOL",
        f"{model} · " + ("相对 BF16 的宏平均 KL · 越低越好" if zh
                          else "macro KL vs aligned BF16 · lower is better"),
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.10, 0.20, 0.50]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # Tier gates: the KL anchors FIT solves to, and the line each APEX point is
    # read against.
    for gate in gates:
        ax.axhline(gate, color=GRID, linewidth=1.1, linestyle=(0, (1, 4)), zorder=1)
        ax.annotate(
            f"{'门' if zh else 'gate'} {gate:.2f}",
            (0.997, gate),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=MUTED,
        )

    ax.scatter(
        [x["gib"] for x in native], [x["kl"] for x in native],
        s=90, marker="D", facecolor=PANEL, edgecolor=BLUE, linewidth=2.0,
        label="llama.cpp 原生预设" if zh else "llama.cpp native presets", zorder=3,
    )
    for item in native:
        dx, dy, ha = NATIVE_LABEL_OFFSETS.get(item["name"], (0, -17, "center"))
        ax.annotate(
            item["name"], (item["gib"], item["kl"]),
            xytext=(dx, dy), textcoords="offset points", ha=ha,
            fontsize=9, color=BLUE,
        )

    ax.plot(
        [x["gib"] for x in fit], [x["kl"] for x in fit],
        color=ORANGE, linewidth=3.2, marker="o", markersize=8.5,
        markeredgecolor=BG, markeredgewidth=1.3,
        label="FIT 档位" if zh else "FIT tiers", zorder=5,
    )
    for item in fit:
        dx, dy, ha = FIT_LABEL_OFFSETS.get(item["key"], (0, 15, "center"))
        ax.annotate(
            f"{item['name']}\n{item['kl']:.4f}", (item["gib"], item["kl"]),
            xytext=(dx, dy), textcoords="offset points", ha=ha,
            fontsize=10.5, color=ORANGE, fontweight="bold", linespacing=1.25,
        )

    ax.plot(
        [x["gib"] for x in apex], [x["kl"] for x in apex],
        color=GREEN, linewidth=2.4, linestyle="--", marker="^", markersize=10,
        markeredgecolor=BG, markeredgewidth=1.3,
        label="APEX I- 档位" if zh else "APEX I- tiers", zorder=4,
    )
    for item in apex:
        dx, dy, ha = APEX_LABEL_OFFSETS.get(item["name"], (0, -30, "center"))
        ax.annotate(
            f"{item['name']}\n{item['kl']:.4f}", (item["gib"], item["kl"]),
            xytext=(dx, dy), textcoords="offset points", ha=ha,
            fontsize=10.5, color=GREEN, fontweight="bold", linespacing=1.25,
        )

    xs = [x["gib"] for x in native + fit + apex]
    ys = [x["kl"] for x in native + fit + apex]
    ax.set_xlim(min(xs) - 0.9, max(xs) + 1.3)
    ax.set_ylim(min(ys) * 0.82, max(ys) * 1.35)
    frame(
        ax,
        xlabel="主 GGUF 文件大小（GiB）· 越低越好（仅指空间占用）" if zh
        else "Main GGUF size (GiB) · lower is better for storage footprint",
        ylabel="五域宏平均 KL 散度 · 越低越好" if zh
        else "Five-domain macro KL divergence · lower is better",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=13, labelcolor=INK)
    # One short line, like the house reference chart. The APEX-variant
    # explanation belongs in the model card, not in a footer: a second sentence
    # here stretches the figure and buys nothing a reader of the chart needs.
    footer(
        fig,
        "协议：llama.cpp b10666 · c=512，b=512 · 五个固定 64 KiB 切片：wiki_test、wiki_valid、中文、代码、agent_chat · 仅为本协议观测。"
        if zh
        else "Protocol: llama.cpp b10666 · c=512, b=512 · five fixed 64 KiB slices: wiki_test, wiki_valid, Chinese, code, agent_chat · observations only.",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    filenames = [f"three-way-{lang}.png"] + (["three-way.png"] if lang == "en" else [])
    for filename in filenames:
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return out_dir / filenames[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, required=True, help="Calibration Bundle (curve-points.jsonl)")
    ap.add_argument("--fit", type=Path, required=True, help="emit-report.json from emit_tier_artifacts.py")
    ap.add_argument("--apex", type=Path, required=True, help="apex-report.json from run_apex_tiers.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lang", default="both", choices=("en", "zh", "both"))
    ap.add_argument("--gates", default="0.05,0.10,0.15,0.20")
    args = ap.parse_args()

    configure()
    native = load_native(args.calibration)
    fit = load_fit(args.fit)
    apex = load_apex(args.apex)
    gates = [float(g) for g in args.gates.split(",") if g.strip()]

    print(f"  native presets : {len(native)}")
    print(f"  FIT tiers      : {len(fit)}")
    print(f"  APEX tiers     : {len(apex)}")
    for lang in (("en", "zh") if args.lang == "both" else (args.lang,)):
        path = render(native, fit, apex, args.model, args.out_dir, lang, gates)
        print(f"  wrote {path}")

    rows = sorted(native + fit + apex, key=lambda r: r["gib"])
    print(f"\n  {'family':>7} {'tier':<10} {'GiB':>7} {'macro KL':>9} {'same-top':>9}")
    for row in rows:
        family = "APEX" if row in apex else ("FIT" if row in fit else "native")
        print(f"  {family:>7} {row['name']:<10} {row['gib']:7.2f} {row['kl']:9.4f} {row['top']:9.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
