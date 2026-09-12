#!/usr/bin/env python3
"""Render the Nex-N2.5-mini-abliterated release charts in the FIT-GGUF house style.

Style constants and layout are lifted from scripts/render_release_charts.py (the
renderer that produced the MiniCPM5 and Qwen3.8 release charts): near-black
canvas, cream ink, orange FIT series, blue native-preset diamonds, log-scale KL,
header/footer framing and leader-line annotations. APEX gets a third accent
(#10B981) from the same brand family used in the release READMEs.

    python scripts/render_nex_release_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/2026-09-11-nex25-mini-4tier"
EVAL = EXP / "apex-vs-fit-eval.json"
CURVE = EXP / "calibration/curve-points.jsonl"
OUT_DIRS = (EXP / "results",)
MODEL = "Nex-N2.5-mini-abliterated"

# --- house palette (identical to render_release_charts.py) ---
BG = "#070707"
PANEL = "#101112"
INK = "#F4F1E8"
MUTED = "#A4A6AA"
GRID = "#313338"
ORANGE = "#FF5A1F"
BLUE = "#5593FF"
GREEN = "#10B981"

# CJK coverage for the zh charts: matplotlib's bundled DejaVu has no Chinese
# glyphs, so register the system Noto CJK faces and keep a fallback chain.
from matplotlib import font_manager as _fm  # noqa: E402

for _ttc in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
):
    try:
        _fm.fontManager.addfont(_ttc)
    except Exception:  # noqa: BLE001 — font is optional, fallback chain covers it
        pass
_CJK = [f.name for f in _fm.fontManager.ttflist if "CJK" in f.name]
mpl.rcParams["font.sans-serif"] = _CJK + ["Droid Sans Fallback", "DejaVu Sans"]

mpl.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.size": 12,
    "axes.unicode_minus": False,
})

NATIVE_SKIP = ("probe-",)
# Labels live in the empty regions (upper-left for the small sizes, open space to
# the right for the large ones) with short leader lines, so no label sits on the
# curves. Offsets are in points from the data point.
FIT_OFFSETS = {
    "MINI": (-96, 34), "COMPACT": (-104, 10), "BALANCED": (-118, -18), "QUALITY": (52, 18),
}
APEX_OFFSETS = {
    "Mini": (-30, 76), "I-Compact": (16, -40), "I-Quality": (-96, 30), "I-Balanced": (54, -34),
}
NATIVE_OFFSETS = {
    "IQ2_XXS": (-30, 14), "IQ2_XS": (-26, 14), "IQ2_M": (-26, 14),
    "IQ3_XXS": (-34, 12), "IQ3_XS": (12, 12), "IQ3_M": (12, 12),
    "Q3_K_M": (-40, -14), "IQ4_XS": (12, -8), "Q4_K_M": (12, 12),
    "Q5_K_M": (12, 12), "Q6_K": (12, -16), "Q8_0": (12, 12),
}


def load() -> tuple[list[dict], list[dict], list[dict]]:
    native, fit, apex = [], [], []
    for line in CURVE.read_text().splitlines():
        p = json.loads(line)
        if any(p["point_id"].startswith(s) for s in NATIVE_SKIP):
            continue
        native.append({"name": p["point_id"], "gib": p["size_bytes"] / 2**30,
                       "kl": p["macro_kl"], "top": p["same_top"] * 100.0})
    for a in json.loads(EVAL.read_text())["artifacts"]:
        name = a["name"]
        if "APEX" in name:
            tier = name.split("APEX-")[-1].replace(".gguf", "")
            apex.append({"name": tier, "gib": a["size_bytes"] / 2**30,
                         "kl": a["macro_kl"], "top": a["same_top_pct"]})
        else:
            tier = name.split("FIT-")[1].split("-")[0]
            fit.append({"name": tier, "gib": a["size_bytes"] / 2**30,
                        "kl": a["macro_kl"], "top": a["same_top_pct"]})
    for group in (native, fit, apex):
        group.sort(key=lambda r: r["gib"])
    return native, fit, apex


def frame(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=16)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, alpha=0.7)
    ax.grid(True, which="minor", color=GRID, linewidth=0.5, alpha=0.28)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.065, 0.94, title, fontsize=30, weight="bold", color=INK)
    fig.text(0.065, 0.902, subtitle, fontsize=15, color=MUTED)
    fig.text(0.935, 0.935, "FIT-GGUF", fontsize=18, weight="bold", color=ORANGE, ha="right")


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.065, 0.018, text, fontsize=11, color=MUTED)


def save(fig: plt.Figure, stem: str, lang: str) -> None:
    names = [f"{stem}-{lang}.png"] + ([f"{stem}.png"] if lang == "en" else [])
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            fig.savefig(directory / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render(lang: str, metric: str) -> None:
    zh = lang == "zh"
    native, fit, apex = load()
    key = "kl" if metric == "kl" else "top"
    logy = metric == "kl"

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.18)

    if metric == "kl":
        header(
            fig,
            "文件尺寸与质量曲线" if zh else "QUALITY ACROSS THE SIZE CURVE",
            f"{MODEL} · 相对 BF16 的宏平均 KL · 越低越好" if zh
            else f"{MODEL} · macro KL vs aligned BF16 · lower is better",
        )
    else:
        header(
            fig,
            "同 top-1 一致率" if zh else "SAME-TOP AGREEMENT",
            f"{MODEL} · 与对齐 BF16 选出同一最高概率 token 的比例 · 越高越好" if zh
            else f"{MODEL} · share of positions agreeing with aligned BF16 · higher is better",
        )

    if logy:
        ax.set_yscale("log")
        ticks = [0.03, 0.05, 0.10, 0.20, 0.50, 1.00]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.scatter([x["gib"] for x in native], [x[key] for x in native], s=90, marker="D",
               facecolor=PANEL, edgecolor=BLUE, linewidth=2.0, zorder=3,
               label="llama.cpp 原生预设" if zh else "llama.cpp presets")

    ax.plot([x["gib"] for x in fit], [x[key] for x in fit], color=ORANGE, linewidth=3.2,
            marker="o", markersize=8.5, markeredgecolor=BG, markeredgewidth=1.3, zorder=4,
            label="FIT 档位（KL 锚定）" if zh else "FIT tiers (KL-anchored)")

    ax.plot([x["gib"] for x in apex], [x[key] for x in apex], color=GREEN, linewidth=3.2,
            marker="^", markersize=9.5, markeredgecolor=BG, markeredgewidth=1.3, zorder=5,
            label="APEX-I 档位" if zh else "APEX-I tiers")

    for item in fit:
        dx, dy = FIT_OFFSETS.get(item["name"], (0, 16))
        ax.annotate(item["name"], (item["gib"], item[key]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11, color=ORANGE, weight="bold",
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.9, "alpha": 0.75})
    for item in apex:
        dx, dy = APEX_OFFSETS.get(item["name"], (0, 16))
        ax.annotate(item["name"], (item["gib"], item[key]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=11, color=GREEN, weight="bold",
                    ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 0.9, "alpha": 0.75})
    for item in native:
        dx, dy = NATIVE_OFFSETS.get(item["name"], (12, 12))
        ax.annotate(item["name"], (item["gib"], item[key]), xytext=(dx, dy),
                    textcoords="offset points", ha="left" if dx > 0 else "right",
                    va="center", fontsize=9.5, color=BLUE,
                    arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7, "alpha": 0.65})

    if metric == "kl":
        # The headline comparison: native's best sub-16 GiB point vs both families.
        ax.annotate(
            ("同尺寸下 KL 低 38%\nFIT 0.0998 / APEX 0.1006\n对原生 Q3_K_M 的 0.1615") if zh
            else ("same bytes, 38% lower KL\nFIT 0.0998 · APEX 0.1006\nvs native Q3_K_M 0.1615"),
            xy=(15.61, 0.1615), xycoords="data", xytext=(17.2, 0.34), textcoords="data",
            fontsize=12.5, color=INK, ha="left", va="center",
            arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.2},
        )
        best = min(apex, key=lambda r: r["kl"])
        ax.scatter([best["gib"]], [best["kl"]], s=330, facecolor="none",
                   edgecolor=GREEN, linewidth=1.6, zorder=6)
        ax.annotate(
            (f"全场每字节最优\n{best['name']} KL {best['kl']:.4f}") if zh
            else (f"best KL per byte measured\n{best['name']} @ {best['kl']:.4f}"),
            (best["gib"], best["kl"]), xycoords="data", xytext=(26.6, 0.052), textcoords="data",
            fontsize=11.5, color=INK,
            arrowprops={"arrowstyle": "-", "color": GREEN, "lw": 1.2},
        )

    frame(
        ax,
        "主 GGUF 体积 (GiB) · 存储占用越低越好" if zh
        else "Main GGUF size (GiB) · lower is better for storage footprint",
        ("五域宏平均 KL 散度 · 越低越好" if zh else "Five-domain macro KL divergence · lower is better")
        if metric == "kl" else
        ("同 top-1 一致率 % · 越高越好" if zh else "Same-top agreement % · higher is better"),
    )
    ax.legend(loc="lower right" if metric != "kl" else "upper right",
              framealpha=0.0, labelcolor=INK, fontsize=11.5)
    if metric == "kl":
        ax.annotate(
            "BF16 参考 64.61 GiB（KL = 0，在本轴下方）" if zh
            else "BF16 reference 64.61 GiB (KL = 0, below this axis)",
            xy=(0.012, 0.03), xycoords="axes fraction", fontsize=11, color=MUTED,
        )
    footer(
        fig,
        "评估协议：llama.cpp b10666（ROCm）· c=512, b=512 · 五个固定 64 KiB 切片：wiki_test, wiki_valid, Chinese, code, agent_chat · "
        "三族（原生 / FIT / APEX）用同一套协议测量。观察结论，非普适质量保证。"
        if zh else
        "Protocol: llama.cpp b10666 (ROCm) · c=512, b=512 · five fixed 64 KiB slices: wiki_test, wiki_valid, Chinese, code, agent_chat · "
        "all three families measured with the identical protocol. Observations, not a universal quality guarantee.",
    )
    save(fig, f"{'kl' if metric == 'kl' else 'sametop'}-curve", lang)


if __name__ == "__main__":
    for language in ("en", "zh"):
        render(language, "kl")
        render(language, "sametop")
    print(f"wrote charts to {OUT_DIRS[0]}")
