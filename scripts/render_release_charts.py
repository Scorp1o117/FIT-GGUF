#!/usr/bin/env python3
"""Render public FIT-GGUF release charts from the consolidated P4 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/2026-08-29-p4-release-batch/results/p4-results.json"
BASELINE_SOURCE = (
    ROOT
    / "experiments/2026-08-30-p5-kfree-12-13.5/results/baseline-k-based/p4-results.json"
)
OUTPUT_DIRS = (
    ROOT / "docs/assets",
    ROOT / "Qwen3.8-27B-Uncensored-FIT-GGUF/results",
)

BG = "#070707"
PANEL = "#101112"
INK = "#F4F1E8"
MUTED = "#A4A6AA"
GRID = "#313338"
ORANGE = "#FF5A1F"
BLUE = "#5593FF"

REF_LABEL_OFFSETS_KL = {
    "IQ1_S": (-48, 18),
    "IQ1_M": (14, 20),
    "IQ2_XXS": (-58, -18),
    "IQ2_XS": (-48, 20),
    "IQ2_S": (14, 18),
    "IQ2_M": (-56, 20),
    "Q2_K_S": (-58, 20),
    "Q2_K": (14, -20),
    "IQ3_XXS": (-68, -24),
    "IQ3_XS": (-68, -24),
    "Q3_K_S": (14, 20),
    "IQ3_S": (14, 22),
    "IQ3_M": (18, -26),
    "IQ4_XS": (14, -8),
}

REF_LABEL_OFFSETS_SAME = {
    "IQ1_S": (-48, -15),
    "IQ1_M": (14, 15),
    "IQ2_XXS": (-62, -18),
    "IQ2_XS": (-54, 20),
    "IQ2_S": (14, -20),
    "IQ2_M": (-54, -20),
    "Q2_K_S": (-58, 20),
    "Q2_K": (14, -20),
    "IQ3_XXS": (-68, 20),
    "IQ3_XS": (-64, 20),
    "Q3_K_S": (-64, -20),
    "IQ3_S": (14, 22),
    "IQ3_M": (18, -24),
    "IQ4_XS": (-64, 16),
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 16,
            "axes.unicode_minus": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
        }
    )


def load() -> tuple[list[dict], list[dict]]:
    artifacts = json.loads(SOURCE.read_text())["artifacts"]
    fits: list[dict] = []
    refs: list[dict] = []
    for name, raw in artifacts.items():
        item = {"name": name, **raw, "gib": raw["actual_bytes"] / 2**30}
        (fits if raw["kind"] == "fit" else refs).append(item)
    fits.sort(key=lambda item: item["gib"])
    refs.sort(key=lambda item: item["gib"])
    return fits, refs


def load_baseline() -> dict[str, dict]:
    return json.loads(BASELINE_SOURCE.read_text())["artifacts"]


def frame(ax: plt.Axes, *, xlabel: str, ylabel: str) -> None:
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
    filenames = [f"{stem}-{lang}.png"]
    if lang == "en":
        filenames.append(f"{stem}.png")
    for directory in OUTPUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            fig.savefig(directory / filename, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render_kl(fits: list[dict], refs: list[dict], lang: str) -> None:
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.18)
    header(
        fig,
        "不同文件尺寸下的质量曲线" if zh else "QUALITY ACROSS THE SIZE CURVE",
        "Qwen3.8-27B-Uncensored · 相对 BF16 的宏平均 KL · 越低越好"
        if zh
        else "Qwen3.8-27B-Uncensored · macro KL vs aligned BF16 · lower is better",
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.10, 0.20, 0.50, 1.00]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.scatter(
        [x["gib"] for x in refs],
        [x["macro_kld"] for x in refs],
        s=90,
        marker="D",
        facecolor=PANEL,
        edgecolor=BLUE,
        linewidth=2.0,
        label="llama.cpp 原生预设" if zh else "llama.cpp presets",
        zorder=3,
    )
    ax.plot(
        [x["gib"] for x in fits],
        [x["macro_kld"] for x in fits],
        color=ORANGE,
        linewidth=3.2,
        marker="o",
        markersize=7.5,
        markeredgecolor=BG,
        markeredgewidth=1.3,
        label="FIT 档位" if zh else "FIT tiers",
        zorder=4,
    )

    for index, item in enumerate(fits):
        dy = 13 if index % 2 == 0 else -20
        if item["name"] in {"FIT-11G", "FIT-11.5G", "FIT-12G"}:
            dy = {"FIT-11G": 12, "FIT-11.5G": -21, "FIT-12G": 14}[item["name"]]
        ax.annotate(
            item["name"].removeprefix("FIT-"),
            (item["gib"], item["macro_kld"]),
            xytext=(0, dy),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=ORANGE,
        )

    for item in refs:
        dx, dy = REF_LABEL_OFFSETS_KL[item["name"]]
        ax.annotate(
            item["name"],
            (item["gib"], item["macro_kld"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=9.5,
            color=BLUE,
            arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7, "alpha": 0.65},
        )

    best = next(x for x in fits if x["name"] == "FIT-13.5G")
    ax.scatter([best["gib"]], [best["macro_kld"]], s=310, facecolor="none", edgecolor=ORANGE, linewidth=1.5)
    ax.annotate(
        "最佳 FIT 档位\nKL 0.0838" if zh else "BEST FIT TIER\nKL 0.0838",
        (best["gib"], best["macro_kld"]),
        xytext=(-108, -45),
        textcoords="offset points",
        fontsize=11,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.2},
    )

    ax.set_xlim(6.45, 14.45)
    ax.set_ylim(0.045, 1.5)
    frame(
        ax,
        xlabel="主 GGUF 文件大小（GiB）· 越低越好（仅指空间占用）"
        if zh
        else "Main GGUF size (GiB) · lower is better for storage footprint",
        ylabel="五域宏平均 KL 散度 · 越低越好"
        if zh
        else "Five-domain macro KL divergence · lower is better",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=13, labelcolor=INK)
    footer(
        fig,
        "协议：llama.cpp b10666 · c=512，b=512 · 五个固定 64 KiB 切片：wiki_test、wiki_valid、中文、代码、agent_chat。仅为本协议观测，不代表通用质量保证。"
        if zh
        else "Protocol: llama.cpp b10666 · c=512, b=512 · five fixed 64 KiB slices: wiki_test, wiki_valid, Chinese, code, agent_chat. Observations, not a universal quality guarantee.",
    )
    save(fig, "kl-curve", lang)


def render_same_top(fits: list[dict], refs: list[dict], lang: str) -> None:
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.18)
    header(
        fig,
        "不同文件尺寸下的首选词元一致率" if zh else "TOP-TOKEN AGREEMENT ACROSS THE CURVE",
        "Qwen3.8-27B-Uncensored · 相对 BF16 的宏平均 Same-top · 越高越好"
        if zh
        else "Qwen3.8-27B-Uncensored · macro Same-top vs aligned BF16 · higher is better",
    )
    ax.scatter(
        [x["gib"] for x in refs],
        [x["macro_same_top"] for x in refs],
        s=90,
        marker="D",
        facecolor=PANEL,
        edgecolor=BLUE,
        linewidth=2.0,
        label="llama.cpp 原生预设" if zh else "llama.cpp presets",
        zorder=3,
    )
    ax.plot(
        [x["gib"] for x in fits],
        [x["macro_same_top"] for x in fits],
        color=ORANGE,
        linewidth=3.2,
        marker="o",
        markersize=7.5,
        markeredgecolor=BG,
        markeredgewidth=1.3,
        label="FIT 档位" if zh else "FIT tiers",
        zorder=4,
    )
    for index, item in enumerate(fits):
        ax.annotate(
            item["name"].removeprefix("FIT-"),
            (item["gib"], item["macro_same_top"]),
            xytext=(0, 12 if index % 2 == 0 else -19),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=ORANGE,
        )
    for item in refs:
        dx, dy = REF_LABEL_OFFSETS_SAME[item["name"]]
        ax.annotate(
            item["name"],
            (item["gib"], item["macro_same_top"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx > 0 else "right",
            va="center",
            fontsize=9.5,
            color=BLUE,
            arrowprops={"arrowstyle": "-", "color": BLUE, "lw": 0.7, "alpha": 0.65},
        )
    ax.set_xlim(6.45, 14.45)
    ax.set_ylim(60, 95.5)
    frame(
        ax,
        xlabel="主 GGUF 文件大小（GiB）· 越低越好（仅指空间占用）"
        if zh
        else "Main GGUF size (GiB) · lower is better for storage footprint",
        ylabel="五域宏平均 Same-top（%）· 越高越好"
        if zh
        else "Five-domain macro Same-top (%) · higher is better",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=13, labelcolor=INK)
    footer(
        fig,
        "Same-top 表示量化模型与对齐 BF16 参考在同一位置选中相同最高 logit 词元的比例。仅为本协议范围内的观测。"
        if zh
        else "Same-top is the share of positions where the quantized model and aligned BF16 reference select the same highest-logit token. Protocol-scoped observation.",
    )
    save(fig, "sametop-curve", lang)


def render_utilization(fits: list[dict], lang: str) -> None:
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.14, right=0.96, top=0.84, bottom=0.18)
    header(
        fig,
        "目标预算利用率" if zh else "TARGET-BUDGET UTILIZATION",
        "实际主 GGUF 文件大小占请求档位预算的比例"
        if zh
        else "Actual main GGUF size as a share of the requested tier budget",
    )
    labels = [x["name"].removeprefix("FIT-") for x in fits]
    utilization = [100 * x["actual_bytes"] / x["target_bytes"] for x in fits]
    slack_mib = [(x["target_bytes"] - x["actual_bytes"]) / 2**20 for x in fits]
    y = list(range(len(fits)))
    colors = [ORANGE if x["name"] != "FIT-11.5G" else BLUE for x in fits]
    ax.barh(y, utilization, left=99.35, color=colors, height=0.55)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for yi, value, slack in zip(y, utilization, slack_mib):
        suffix = f"{slack:.1f} MiB 空余" if zh else f"{slack:.1f} MiB slack"
        ax.text(value + 0.008, yi, f"{value:.3f}%  ·  {suffix}", va="center", fontsize=11, color=INK)
    ax.axvline(100, color=INK, linewidth=1.2, alpha=0.7)
    ax.set_xlim(99.35, 100.08)
    ax.set_xticks([99.4, 99.6, 99.8, 100.0])
    frame(
        ax,
        xlabel="实际大小 / 请求目标（%）· 越高越好（越接近请求预算）"
        if zh
        else "Actual / requested target (%) · higher is better (closer to target)",
        ylabel="FIT 档位 · 由上到下预算越高" if zh else "FIT tier · budget increases downward",
    )
    ax.text(
        0.99,
        0.02,
        "蓝色：FIT-11.5G 的 oracle/计数规则空余（已记录）"
        if zh
        else "Blue: FIT-11.5G oracle/counter-rule slack (documented)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=BLUE,
        fontsize=11,
    )
    footer(
        fig,
        "全部 14 个实际产物均与 oracle 后预测字节数精确一致。目标空余来自可表示 Tensor/qtype 配方空间的离散性，不属于预测误差。"
        if zh
        else "All 14 actual artifacts matched their post-oracle predicted byte sizes exactly. Target slack reflects the discrete representable tensor/qtype recipe space; it is not prediction error.",
    )
    save(fig, "target-utilization", lang)


def render_strategy_improvement(fits: list[dict], baseline: dict[str, dict], lang: str) -> None:
    zh = lang == "zh"
    names = [
        "FIT-8G",
        "FIT-8.5G",
        "FIT-9G",
        "FIT-9.5G",
        "FIT-10G",
        "FIT-12.5G",
        "FIT-13G",
        "FIT-13.5G",
    ]
    current = {item["name"]: item for item in fits}
    old_kl = [baseline[name]["macro_kld"] for name in names]
    new_kl = [current[name]["macro_kld"] for name in names]
    old_same = [baseline[name]["macro_same_top"] for name in names]
    new_same = [current[name]["macro_same_top"] for name in names]
    y = list(range(len(names)))
    height = 0.32

    fig, (ax_kl, ax_same) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"wspace": 0.20})
    fig.subplots_adjust(left=0.10, right=0.96, top=0.80, bottom=0.17)
    header(
        fig,
        "分配策略修复前后对比" if zh else "ALLOCATION STRATEGY REPAIR",
        "八个替换档位 · 保留的旧发布候选 vs 最终 P5/P6 配方"
        if zh
        else "Eight replaced FIT tiers · retained old release candidates vs final P5/P6 recipes",
    )

    ax_kl.barh([v + height / 2 for v in y], old_kl, height=height, color=BLUE, alpha=0.42, label="旧配方" if zh else "Old recipe")
    ax_kl.barh([v - height / 2 for v in y], new_kl, height=height, color=ORANGE, label="最终配方" if zh else "Final recipe")
    for yi, old, new in zip(y, old_kl, new_kl):
        ax_kl.text(old + 0.008, yi + height / 2, f"{old:.4f}", va="center", fontsize=10, color=BLUE)
        ax_kl.text(new + 0.008, yi - height / 2, f"{new:.4f}", va="center", fontsize=10, color=INK)
    ax_kl.set_yticks(y, [name.removeprefix("FIT-") for name in names])
    ax_kl.invert_yaxis()
    ax_kl.set_xlim(0, 0.64)
    ax_kl.set_title("宏平均 KL 散度 ↓" if zh else "Macro KL divergence ↓", loc="left", fontsize=18, pad=16)
    frame(
        ax_kl,
        xlabel="越低越好" if zh else "Lower is better",
        ylabel="FIT 档位 · 由上到下预算越高" if zh else "FIT tier · budget increases downward",
    )
    ax_kl.legend(loc="lower right", frameon=False, fontsize=12, labelcolor=INK)

    ax_same.barh([v + height / 2 for v in y], old_same, height=height, color=BLUE, alpha=0.42, label="旧配方" if zh else "Old recipe")
    ax_same.barh([v - height / 2 for v in y], new_same, height=height, color=ORANGE, label="最终配方" if zh else "Final recipe")
    for yi, old, new in zip(y, old_same, new_same):
        ax_same.text(old + 0.16, yi + height / 2, f"{old:.1f}%", va="center", fontsize=10, color=BLUE)
        ax_same.text(new + 0.16, yi - height / 2, f"{new:.1f}%", va="center", fontsize=10, color=INK)
    ax_same.set_yticks(y, [])
    ax_same.invert_yaxis()
    ax_same.set_xlim(72, 95)
    ax_same.set_title("Same-top 一致率 ↑" if zh else "Same-top agreement ↑", loc="left", fontsize=18, pad=16)
    frame(ax_same, xlabel="越高越好（%）" if zh else "Higher is better (%)", ylabel="")

    footer(
        fig,
        "旧值：保留的修复前 P4 候选。最终值：合并后的 P5 K-free 与 P6 IQ2 跨度替换产物。模型、BF16 参考、切片及评测协议均相同。"
        if zh
        else "Old values: retained pre-repair P4 candidates. Final values: consolidated P5 K-free and P6 IQ2-span replacements. Same model, BF16 reference, slices and evaluation protocol.",
    )
    save(fig, "strategy-improvement", lang)


def main() -> None:
    configure()
    fits, refs = load()
    baseline = load_baseline()
    assert len(fits) == 14 and len(refs) == 14
    for lang in ("en", "zh"):
        render_kl(fits, refs, lang)
        render_same_top(fits, refs, lang)
        render_utilization(fits, lang)
        render_strategy_improvement(fits, baseline, lang)


if __name__ == "__main__":
    main()
