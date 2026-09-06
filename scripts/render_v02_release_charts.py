#!/usr/bin/env python3
"""Render the public FIT-GGUF v0.2 result charts from release evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from render_release_charts import (
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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/2026-09-02-m2-topkl-calibration"
OUTPUT_DIR = ROOT / "docs/assets"

TIERS = (
    {
        "id": "quality",
        "label": "Quality",
        "kl_limit": 0.05,
        "top_floor": 94.75,
        "smaller": ("Q4_K_M", 15.41, 0.0658, 93.79, "FAIL-BOTH"),
        "larger": ("Q5_K_S", 17.40, 0.0455, 95.59, "PASS"),
        "saving": 6.6,
    },
    {
        "id": "balanced",
        "label": "Balanced",
        "kl_limit": 0.10,
        "top_floor": 91.18,
        "smaller": ("IQ3_M", 11.72, 0.1445, 89.38, "FAIL-BOTH"),
        "larger": ("IQ4_XS", 14.05, 0.0624, 93.79, "PASS"),
        "saving": 8.6,
    },
    {
        "id": "compact",
        "label": "Compact",
        "kl_limit": 0.15,
        "top_floor": 88.94,
        "smaller": ("IQ3_XS", 11.15, 0.1512, 89.05, "FAIL-KL"),
        "larger": ("IQ3_S", 11.57, 0.1424, 89.53, "PASS"),
        "saving": 3.4,
    },
    {
        "id": "mini",
        "label": "Mini",
        "kl_limit": 0.20,
        "top_floor": 85.03,
        "smaller": ("Q2_K", 9.98, 0.2439, 84.08, "FAIL-BOTH"),
        "larger": ("IQ3_XXS", 10.42, 0.1946, 87.12, "PASS"),
        "saving": 0.6,
    },
)


def load_release_data() -> tuple[list[dict], list[dict]]:
    release: list[dict] = []
    for tier in TIERS:
        tier_id = tier["id"]
        if tier_id == "compact":
            path = EXPERIMENT / "results-fs/compact-adopt/promotion-record.json"
            raw = json.loads(path.read_text())
            best = {
                "size_bytes": raw["size_bytes"],
                "macro_kl": raw["final_body_verify_eval"]["macro_kl"],
                "same_top": raw["final_body_verify_eval"]["macro_same_top_pct"] / 100,
                "passed": raw["fit.fidelity.final_artifact_verified"],
            }
        else:
            path = (
                EXPERIMENT
                / f"results-fs/{tier_id}/fidelity-search-{tier_id}-summary.json"
            )
            best = json.loads(path.read_text())["best"]
        assert best["passed"] is True
        release.append({**tier, "fit": best})

    points_by_size: dict[int, dict] = {}
    evidence = EXPERIMENT / "results-fs/compact-adopt/fidelity-search-compact.jsonl"
    wanted_sizes = {
        11991706848,
        12016037088,
        12023164128,
        12038647008,
        12072070368,
        12076248288,
    }
    for line in evidence.read_text().splitlines():
        item = json.loads(line)
        if item.get("event") == "seed" and item.get("size_bytes") in wanted_sizes:
            points_by_size[item["size_bytes"]] = item
    sweep = json.loads((EXPERIMENT / "results/r2-reference-sweep.json").read_text())
    for item in sweep["tiers"]["compact"]["grid"]:
        if item["delivered"] in wanted_sizes:
            points_by_size[item["delivered"]] = {
                "size_bytes": item["delivered"],
                "macro_kl": item["macro_kl"],
                "same_top": item["same_top_pct"] / 100,
                "passed": item["passed"],
                "source": item["source"],
            }
    assert set(points_by_size) == wanted_sizes
    compact_points = [points_by_size[size] for size in sorted(points_by_size)]
    return release, compact_points


def save(fig: plt.Figure, stem: str, lang: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = [f"{stem}-{lang}.png"]
    if lang == "en":
        filenames.append(f"{stem}.png")
    for filename in filenames:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def render_minimum_size(rows: list[dict], lang: str) -> None:
    zh = lang == "zh"
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.18)
    header(
        fig,
        "固定保真度下的最小验证体积" if zh else "MINIMUM VERIFIED SIZE AT FIXED FIDELITY",
        "四档 Fidelity Contract · 较小预设失败，FIT 在离散台阶之间停止"
        if zh
        else "Four Fidelity Contracts · smaller preset fails; FIT stops between discrete steps",
    )

    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, rows):
        smaller_name, smaller_gib, _, _, smaller_status = row["smaller"]
        larger_name, larger_gib, _, _, _ = row["larger"]
        fit_gib = row["fit"]["size_bytes"] / 2**30
        ax.plot([smaller_gib, larger_gib], [y, y], color=GRID, linewidth=4, zorder=1)
        ax.scatter(smaller_gib, y, s=160, marker="X", color=MUTED, zorder=3)
        ax.scatter(
            larger_gib,
            y,
            s=150,
            marker="D",
            facecolor=PANEL,
            edgecolor=BLUE,
            linewidth=2.4,
            zorder=3,
        )
        ax.scatter(
            fit_gib,
            y,
            s=245,
            marker="o",
            color=ORANGE,
            edgecolor=BG,
            linewidth=1.8,
            zorder=4,
        )
        smaller_dx = -34 if row["id"] in {"compact", "mini"} else 0
        larger_dx = 34 if row["id"] in {"compact", "mini"} else 0
        fit_dx = -28 if row["id"] in {"compact", "mini"} else 0
        ax.annotate(
            f"{smaller_name}\n{smaller_gib:.2f} GiB · {smaller_status}",
            (smaller_gib, y),
            xytext=(smaller_dx, -42),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            color=MUTED,
        )
        ax.annotate(
            f"FIT v0.2\n{fit_gib:.2f} GiB · PASS",
            (fit_gib, y),
            xytext=(fit_dx, 22),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
            color=ORANGE,
        )
        ax.annotate(
            f"{larger_name}\n{larger_gib:.2f} GiB · PASS",
            (larger_gib, y),
            xytext=(larger_dx, -42),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            color=BLUE,
        )
        ax.annotate(
            f"−{row['saving']:.1f}%",
            xy=(larger_gib, y + 0.22),
            xytext=(fit_gib, y + 0.22),
            arrowprops={"arrowstyle": "<->", "color": INK, "lw": 1.1},
            ha="center",
            va="bottom",
            color=INK,
            fontsize=12,
            weight="bold",
        )

    ax.set_yticks(y_positions, [row["label"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(9.3, 18.05)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    frame(
        ax,
        xlabel="主 GGUF 文件大小（GiB）· 越低越好" if zh else "Main GGUF size (GiB) · lower is better",
        ylabel="保真度档位" if zh else "Fidelity tier",
    )
    legend = [
        Line2D([], [], marker="X", linestyle="none", color=MUTED, markersize=11, label="较小预设：FAIL" if zh else "Smaller preset: FAIL"),
        Line2D([], [], marker="o", linestyle="none", color=ORANGE, markersize=11, label="FIT v0.2：最小验证 PASS" if zh else "FIT v0.2: minimum verified PASS"),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=PANEL, markeredgecolor=BLUE, markersize=10, label="较大预设：PASS" if zh else "Larger preset: PASS"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=11, labelcolor=INK)
    footer(
        fig,
        "旗舰案例：orcarouter/Qwen3.8-27B-Uncensored。节省量相对最近的较大 PASS 预设；仅限已验证 healthy frontier，搜索容差 128 MiB。"
        if zh
        else "Flagship case: orcarouter/Qwen3.8-27B-Uncensored. Savings vs nearest larger PASS preset; validated healthy frontier only, 128 MiB tolerance.",
    )
    save(fig, "v02-minimum-verified-size", lang)


def render_gate_margins(rows: list[dict], lang: str) -> None:
    zh = lang == "zh"
    labels = [row["label"] for row in rows]
    y = list(range(len(rows)))
    kl_use = [100 * row["fit"]["macro_kl"] / row["kl_limit"] for row in rows]
    top_margin = [100 * row["fit"]["same_top"] - row["top_floor"] for row in rows]

    fig, (ax_kl, ax_top) = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"wspace": 0.22})
    fig.subplots_adjust(left=0.10, right=0.96, top=0.80, bottom=0.18)
    header(
        fig,
        "四档双硬门全部通过" if zh else "ALL FOUR TIERS PASS BOTH HARD GATES",
        "KL 上限 ∧ 模型专属 Same-top Guard · 四档主动约束均为 KL"
        if zh
        else "KL ceiling ∧ model-specific Same-top Guard · KL is active in every tier",
    )

    ax_kl.barh(y, kl_use, color=ORANGE, height=0.55)
    ax_kl.axvline(100, color=INK, linewidth=1.4, linestyle="--")
    for yi, value, row in zip(y, kl_use, rows):
        ax_kl.text(value - 0.3, yi, f"{row['fit']['macro_kl']:.4f} / {row['kl_limit']:.2f}", ha="right", va="center", fontsize=11, color=BG, weight="bold")
        ax_kl.text(100.25, yi, "PASS", va="center", fontsize=10, color=ORANGE, weight="bold")
    ax_kl.set_yticks(y, labels)
    ax_kl.invert_yaxis()
    ax_kl.set_xlim(93.5, 102.2)
    ax_kl.set_title("KL 预算占用 ↓" if zh else "KL budget used ↓", loc="left", fontsize=18, pad=16)
    frame(ax_kl, xlabel="实测 KL / 档位上限（%）" if zh else "Measured KL / tier ceiling (%)", ylabel="保真度档位" if zh else "Fidelity tier")

    ax_top.barh(y, top_margin, color=BLUE, height=0.55)
    ax_top.axvline(0, color=INK, linewidth=1.4, linestyle="--")
    for yi, value, row in zip(y, top_margin, rows):
        ax_top.text(value + 0.025, yi, f"+{value:.2f} pp", va="center", fontsize=11, color=INK, weight="bold")
        ax_top.text(-0.02, yi, f"{100 * row['fit']['same_top']:.2f}%", ha="right", va="center", fontsize=10, color=BLUE)
    ax_top.set_yticks(y, [])
    ax_top.invert_yaxis()
    ax_top.set_xlim(-0.25, 1.48)
    ax_top.set_title("Same-top Guard 余量 ↑" if zh else "Same-top Guard margin ↑", loc="left", fontsize=18, pad=16)
    frame(ax_top, xlabel="高于模型专属 Guard floor（百分点）" if zh else "Above model-specific Guard floor (percentage points)", ylabel="")

    footer(
        fig,
        "PASS = 宏平均 KL ≤ 档位上限 且 Same-top ≥ 已验证模型专属 Guard。数值来自最终发布工件评测。"
        if zh
        else "PASS = macro KL ≤ tier ceiling and Same-top ≥ validated model-specific Guard. Values come from final release-artifact evaluations.",
    )
    save(fig, "v02-fidelity-gates", lang)


def render_compact_inversion(points: list[dict], lang: str) -> None:
    zh = lang == "zh"
    x = [item["size_bytes"] / 2**30 for item in points]
    y = [item["macro_kl"] for item in points]

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.82, bottom=0.18)
    header(
        fig,
        "为什么最终工件必须实测" if zh else "WHY THE FINAL ARTIFACT MUST BE VERIFIED",
        "Compact 穿越区 · 更大的混合量化配方不保证更好"
        if zh
        else "Compact crossing region · a larger mixed-quant recipe is not guaranteed to be better",
    )
    ax.plot(x, y, color=ORANGE, linewidth=2.4, alpha=0.72, linestyle="--", zorder=2)
    ax.axhspan(0.145, 0.15, color=ORANGE, alpha=0.08)
    ax.axhline(0.15, color=INK, linewidth=1.5, linestyle="--", label="KL 上限 0.1500" if zh else "KL ceiling 0.1500")

    for index, item in enumerate(points):
        gib = item["size_bytes"] / 2**30
        passed = item["passed"]
        marker = "o" if passed else "X"
        color = ORANGE if passed else BLUE
        size = 300 if index == 0 else 190
        ax.scatter(gib, item["macro_kl"], s=size, marker=marker, color=color, edgecolor=BG, linewidth=1.7, zorder=4)
        label_offsets = {
            0: (-8, 34),
            1: (-20, 28),
            2: (18, 24),
            3: (0, -34),
            4: (0, 24),
            5: (0, -34),
        }
        dx, dy = label_offsets[index]
        status = "PASS" if passed else "FAIL"
        label = f"{gib:.2f} GiB\nKL {item['macro_kl']:.4f} · {status}"
        if index == 0:
            label += "\n" + ("发布工件" if zh else "RELEASE ARTIFACT")
        ax.annotate(
            label,
            (gib, item["macro_kl"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy > 0 else "top",
            fontsize=10.5,
            color=color,
            weight="bold" if index == 0 else "normal",
            bbox={"boxstyle": "round,pad=0.24", "fc": PANEL, "ec": color, "lw": 0.7, "alpha": 0.94},
        )

    ax.set_xlim(min(x) - 0.012, max(x) + 0.012)
    ax.set_ylim(0.1468, 0.1602)
    frame(
        ax,
        xlabel="主 GGUF 文件大小（GiB）" if zh else "Main GGUF size (GiB)",
        ylabel="五域宏平均 KL · 越低越好" if zh else "Five-domain macro KL · lower is better",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=12, labelcolor=INK)
    footer(
        fig,
        "CLI 将该交叉判为 noise_inversion 并拒绝自动交付；11.17 GiB 工件经重建、exact-byte 校验和最终本体复评后人工晋升。"
        if zh
        else "CLI classified this crossing as noise_inversion and withheld auto-delivery; the 11.17 GiB artifact was rebuilt, exact-byte checked, re-evaluated, then manually promoted.",
    )
    save(fig, "v02-compact-noise-inversion", lang)


def render_release_gates(lang: str) -> None:
    zh = lang == "zh"
    gates = [
        ("R1", "保真度正确性" if zh else "Fidelity correctness"),
        ("R2", "搜索准确性" if zh else "Search accuracy"),
        ("R3", "搜索预算" if zh else "Search budget"),
        ("R4", "精确字节保证" if zh else "Exact-byte guarantee"),
        ("R5", "v0.1 非回退" if zh else "v0.1 non-regression"),
        ("R6", "可复现性" if zh else "Reproducibility"),
    ]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.065, right=0.96, top=0.82, bottom=0.15)
    header(
        fig,
        "V0.2 发布验证" if zh else "V0.2 RELEASE VALIDATION",
        "旗舰模型发布门 · 修正后的发布验证全部通过"
        if zh
        else "Flagship-model release gates · amended release validation passes in full",
    )
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (gate_id, label) in enumerate(gates):
        col = index % 2
        row = 2 - index // 2
        x0 = 0.06 + col * 1.0
        y0 = row + 0.14
        box = FancyBboxPatch(
            (x0, y0),
            0.87,
            0.68,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(x0 + 0.06, y0 + 0.46, gate_id, color=ORANGE, fontsize=16, weight="bold", va="center")
        ax.text(x0 + 0.06, y0 + 0.23, label, color=INK, fontsize=18, weight="bold", va="center")
        ax.text(x0 + 0.80, y0 + 0.34, "PASS", color=ORANGE, fontsize=15, weight="bold", ha="right", va="center")
    fig.text(0.50, 0.105, "6 / 6  PASS", ha="center", color=ORANGE, fontsize=28, weight="bold")
    footer(
        fig,
        "R2 口径：INITIAL FAIL → CORRECTIVE ACTION TAKEN → AMENDED RELEASE VALIDATION PASS；失败历史保留，非 untouched independent sweep。"
        if zh
        else "R2 wording: INITIAL FAIL → CORRECTIVE ACTION TAKEN → AMENDED RELEASE VALIDATION PASS; failed history retained, not an untouched independent sweep.",
    )
    save(fig, "v02-release-gates", lang)


def render_contact_sheet(lang: str) -> None:
    stems = (
        "v02-minimum-verified-size",
        "v02-fidelity-gates",
        "v02-compact-noise-inversion",
        "v02-release-gates",
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.003, right=0.997, top=0.995, bottom=0.005, wspace=0.025, hspace=0.025)
    for ax, stem in zip(axes.flat, stems):
        ax.imshow(plt.imread(OUTPUT_DIR / f"{stem}-{lang}.png"))
        ax.axis("off")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = [f"v02-contact-sheet-{lang}.png"]
    if lang == "en":
        filenames.append("v02-contact-sheet.png")
    for filename in filenames:
        fig.savefig(OUTPUT_DIR / filename, dpi=150, facecolor=BG)
    plt.close(fig)


def main() -> None:
    configure()
    rows, compact_points = load_release_data()
    for lang in ("en", "zh"):
        render_minimum_size(rows, lang)
        render_gate_margins(rows, lang)
        render_compact_inversion(compact_points, lang)
        render_release_gates(lang)
        render_contact_sheet(lang)


if __name__ == "__main__":
    main()
