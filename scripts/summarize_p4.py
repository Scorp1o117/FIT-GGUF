#!/usr/bin/env python3
"""Parse P4 KL eval logs into p4-results.json and render tables + charts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO = Path("/run/media/s117/OS/FIT-GGUF")
P4 = REPO / "experiments/2026-08-29-p4-release-batch"
LOGS = P4 / "artifacts/logs"
RESULTS = P4 / "results"

PAIR_PATTERNS = {
    "ppl_q": re.compile(r"^Mean PPL\(Q\)\s+:\s+([0-9.]+) ±\s+([0-9.]+)$", re.MULTILINE),
    "mean_kld": re.compile(r"^Mean\s+KLD:\s+([0-9.]+) ±\s+([0-9.]+)$", re.MULTILINE),
    "same_top": re.compile(r"^Same top p:\s+([0-9.]+) ±\s+([0-9.]+) %$", re.MULTILINE),
}
DOMAINS = ("wiki_test", "wiki_valid", "chinese", "code", "agent_chat")


def parse_eval_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    result: dict[str, float] = {}
    for name, pattern in PAIR_PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"missing {name} in {path}")
        result[name] = float(match.group(1))
    return result


def parse_ref_log(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", text)
    if match is None:
        raise ValueError(f"incomplete reference log: {path}")
    return {"ppl": float(match.group(1)), "ppl_uncertainty": float(match.group(2))}


def macro(values: dict[str, dict[str, float]], key: str) -> float:
    return sum(values[d][key] for d in DOMAINS) / len(DOMAINS)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    bf16_references = {d: parse_ref_log(LOGS / f"ref-bf16-{d}.log") for d in DOMAINS}

    artifacts: dict[str, dict[str, object]] = {}

    for line in (P4 / "tiers.csv").read_text(encoding="utf-8").splitlines()[1:]:
        tier, lower, upper, target = line.split(",")
        plan = json.loads((P4 / f"tiers/{tier}/fit-plan.json").read_text(encoding="utf-8"))
        record = json.loads(
            (REPO / f"artifacts/fit/release/{plan['suggested_filename']}.quantize-record.json")
            .read_text(encoding="utf-8")
        )
        domains = {
            d: parse_eval_log(LOGS / f"eval-{tier}-{d}.log") for d in DOMAINS
        }
        artifacts[tier] = {
            "kind": "fit",
            "pair": f"{lower}->{upper}",
            "target_bytes": int(target),
            "actual_bytes": record["size_bytes"],
            "dominant_qtype": plan["dominant_qtype"],
            "suggested_filename": plan["suggested_filename"],
            "domains": domains,
            "macro_kld": macro(domains, "mean_kld"),
            "macro_same_top": macro(domains, "same_top"),
        }

    for line in (P4 / "refs.csv").read_text(encoding="utf-8").splitlines()[1:]:
        preset = line.strip()
        ladder = json.loads(
            (REPO / "experiments/2026-08-29-p2-full-envelope/preset-ladder.json")
            .read_text(encoding="utf-8")
        )
        domains = {
            d: parse_eval_log(LOGS / f"eval-ref-{preset}-{d}.log") for d in DOMAINS
        }
        artifacts[preset] = {
            "kind": "reference",
            "target_bytes": ladder["ladder"][preset]["predicted_size_bytes"],
            "actual_bytes": (REPO / f"artifacts/fit/release/refs/{preset}.gguf").stat().st_size,
            "domains": domains,
            "macro_kld": macro(domains, "mean_kld"),
            "macro_same_top": macro(domains, "same_top"),
        }

    payload = {
        "schema_version": 1,
        "protocol": "llama-perplexity -ngl 99 -t 16 -c 512 -b 512 --kl-divergence (M9 slices)",
        "bf16_references": bf16_references,
        "artifacts": artifacts,
    }
    (RESULTS / "p4-results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"parsed {len(artifacts)} artifacts -> results/p4-results.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fits = sorted(
        ((a["actual_bytes"], label, a) for label, a in artifacts.items() if a["kind"] == "fit"),
    )
    refs = sorted(
        ((a["actual_bytes"], label, a) for label, a in artifacts.items() if a["kind"] == "reference"),
    )
    fit_x = [b / 2**30 for b, _, _ in fits]
    ref_x = [b / 2**30 for b, _, _ in refs]

    # Comparison table
    lines = [
        "# P4 Release Batch: KL Comparison",
        "",
        "Protocol: pinned llama-perplexity b10666, -ngl 99 -t 16 -c 512 -b 512,",
        "KL divergence vs aligned BF16 on the five fixed M9 64 KiB slices.",
        "Macro = unweighted mean over the five domains. Lower KL is better;",
        "higher Same-top is better.",
        "",
        "## FIT tiers (balanced allocator, v0.1b)",
        "",
        "| Tier | Pair | Target GiB | Actual GiB | Dominant qtype | Macro KLD | Macro Same-top % |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for size, label, a in fits:
        lines.append(
            f"| {label} | {a['pair']} | {int(a['target_bytes']) / 2**30:g} "
            f"| {size / 2**30:.3f} | {str(a['dominant_qtype']).upper()} "
            f"| {a['macro_kld']:.6f} | {a['macro_same_top']:.3f} |"
        )
    lines += [
        "",
        "## llama.cpp default presets (reference points)",
        "",
        "| Preset | Actual GiB | Macro KLD | Macro Same-top % |",
        "| --- | ---: | ---: | ---: |",
    ]
    for size, label, a in refs:
        lines.append(
            f"| {label} | {size / 2**30:.3f} | {a['macro_kld']:.6f} | {a['macro_same_top']:.3f} |"
        )
    lines += [
        "",
        "## Per-domain mean KLD (all artifacts, ordered by size)",
        "",
        "| Artifact | GiB | wiki_test | wiki_valid | chinese | code | agent_chat | Macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size, label, a in sorted(fits + refs):
        d = a["domains"]
        lines.append(
            f"| {label} | {size / 2**30:.3f} "
            + " ".join(f"| {d[dom]['mean_kld']:.6f}" for dom in DOMAINS)
            + f" | {a['macro_kld']:.6f} |"
        )
    (RESULTS / "comparison-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote results/comparison-table.md")

    # KL curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fit_x, [a["macro_kld"] for _, _, a in fits], "o-", color="#d62728",
            label="FIT tiers (balanced v0.1b)", linewidth=2, markersize=6, zorder=3)
    ax.scatter(ref_x, [a["macro_kld"] for _, _, a in refs], color="#1f77b4", marker="s",
               s=55, label="llama.cpp default presets", zorder=2)
    for size, label, a in refs:
        ax.annotate(label, (size / 2**30, a["macro_kld"]), textcoords="offset points",
                    xytext=(6, -10), fontsize=7, color="#1f77b4")
    for size, label, _ in fits:
        ax.annotate(label.replace("FIT-", ""), (size / 2**30, dict((l, a) for _, l, a in fits)[label]["macro_kld"]),
                    textcoords="offset points", xytext=(-14, 8), fontsize=7, color="#d62728")
    ax.set_xlabel("Artifact size (GiB)")
    ax.set_ylabel("Macro mean KL divergence (lower is better)")
    ax.set_yscale("log")
    ax.set_title("orcarouter/Qwen3.8-27B-Uncensored: FIT tiers vs default presets")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "kl-curve.png", dpi=150)
    plt.close(fig)

    # Same-top curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fit_x, [a["macro_same_top"] for _, _, a in fits], "o-", color="#d62728",
            label="FIT tiers (balanced v0.1b)", linewidth=2, markersize=6, zorder=3)
    ax.scatter(ref_x, [a["macro_same_top"] for _, _, a in refs], color="#1f77b4", marker="s",
               s=55, label="llama.cpp default presets", zorder=2)
    ax.set_xlabel("Artifact size (GiB)")
    ax.set_ylabel("Macro Same-top % (higher is better)")
    ax.set_title("orcarouter/Qwen3.8-27B-Uncensored: Same-top vs size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "sametop-curve.png", dpi=150)
    plt.close(fig)
    print("wrote results/kl-curve.png and results/sametop-curve.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
