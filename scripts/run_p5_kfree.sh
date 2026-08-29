#!/usr/bin/env bash
# P5: K-free redo of FIT-12.5G / 13G / 13.5G (owner directive 2026-08-30).
# plans -> quantizes -> KL evals -> gates K1-K5 -> (on pass) release swap ->
# re-summarize -> P4 gate re-run. A gate failure stops BEFORE the swap.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src

P5=experiments/2026-08-30-p5-kfree-12-13.5
P4=experiments/2026-08-29-p4-release-batch
BUNDLE=Qwen3.8-27B-Uncensored-FIT-GGUF
RT=tools/llama-b10666-rocm
ANALYSIS=$P4/tiers/_analysis/IQ3_M-IQ4_XS/analysis.json
LOGDIR=$P4/artifacts/logs
REFDIR=$P4/artifacts/reference-kld
EVAL=/run/media/s117/OS/Models/eval-data
MODEL=Qwen3.8-27B-Uncensored
mkdir -p "$P5/tiers" "$P5/results/baseline-k-based" "$LOGDIR"

free_kb=$(df --output=avail /run/media/s117/OS | tail -1)
(( free_kb / 1048576 >= 60 )) || { echo "LOW DISK: ${free_kb}kB free" >&2; exit 1; }

# 0. baseline snapshot of the K-based results (before any overwrite)
cp -n "$P4/results/p4-results.json" "$P5/results/baseline-k-based/"
cp -n "$P4/results/comparison-table.md" "$P5/results/baseline-k-based/"

declare -A SLICES=(
  [wiki_test]="$EVAL/kl-eval-64k.txt"
  [wiki_valid]="$EVAL/kl-eval-valid-64k.txt"
  [chinese]="$EVAL/kl-eval-cn-64k.txt"
  [code]="$EVAL/kl-eval-code-64k.txt"
  [agent_chat]="$EVAL/kl-eval-agent-64k.txt"
)

plan_tier () { # tier target
  local tier=$1 target=$2
  if [[ ! -f "$P5/tiers/$tier/fit-plan.json" ]]; then
    echo "== plan $tier (target $target) =="
    python3 -m fit_gguf.cli plan --analysis "$ANALYSIS" \
      --target-bytes "$target" --policy balanced \
      --model-name "$MODEL" --out-prefix "$P5/tiers/$tier/fit" \
      > "$P5/tiers/$tier-plan.log" 2>&1
  fi
  grep -E "dominant|filename|predicted" "$P5/tiers/$tier-plan.log" || tail -3 "$P5/tiers/$tier-plan.log"
}

quantize_tier () { # tier
  local tier=$1
  local plan="$P5/tiers/$tier/fit-plan.json"
  local out="$BUNDLE/$(python3 -c "
import json;print(json.load(open('$plan'))['suggested_filename'])")"
  local expect analysis
  expect=$(python3 -c "
import json;print(json.load(open('$plan'))['predicted_size_bytes'])")
  analysis=$(python3 -c "
import json;print(json.load(open('$plan'))['analysis_path'])")
  if [[ ! -f "$out" ]]; then
    echo "== quantize $tier -> $out (expect $expect) =="
    python3 -m fit_gguf.cli quantize --analysis "$analysis" \
      --tensor-types "$P5/tiers/$tier/fit-tensor-types.txt" \
      --out "$out" --expect-bytes "$expect" \
      > "$P5/tiers/$tier-quantize.log" 2>&1
  fi
  sha256sum "$out" | tee "$P5/tiers/$tier/artifact-sha256.txt"
}

evaluate_tier () { # tier
  local tier=$1
  local plan="$P5/tiers/$tier/fit-plan.json"
  local out="$BUNDLE/$(python3 -c "
import json;print(json.load(open('$plan'))['suggested_filename'])")"
  local sha sent
  sha=$(sha256sum "$out" | cut -d" " -f1)
  for d in wiki_test wiki_valid chinese code agent_chat; do
    local log="$LOGDIR/eval-$tier-$d.log"
    sent="$LOGDIR/eval-$tier-$d.sha"
    # Eval logs are keyed by tier label, which this experiment REUSES from the
    # K-based run; only trust a log whose sentinel matches the artifact sha.
    if [[ ! -f "$log" || ! -f "$sent" || "$(cat "$sent")" != "$sha" ]]; then
      rm -f "$log"
      echo "== eval $tier $d =="
      "$RT/llama-perplexity" -m "$out" -f "${SLICES[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFDIR/bf16-$d.kld" > "$log" 2>&1
      echo "$sha" > "$sent"
    fi
  done
}

# Stage A: plans
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  plan_tier "$tier" "$target"
done < "$P5/tiers.csv"

# Stage B: quantizes
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  quantize_tier "$tier"
done < "$P5/tiers.csv"

# Stage C: evals
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  evaluate_tier "$tier"
done < "$P5/tiers.csv"

# Gates K1-K5 (exit stops the script before the swap on any failure)
python3 scripts/evaluate_p5_gates.py

# Release swap (only reached with ALL_PASS=True)
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  old=$(python3 -c "
import json;print(json.load(open('$P5/results/baseline-k-based/p4-results.json'))['artifacts']['$tier']['suggested_filename'])")
  mkdir -p "$P4/tiers/$tier/k-based" "$BUNDLE/fit-plans/$tier"
  for f in fit-plan.json fit-recipe.json fit-tensor-types.txt fit-oracle-dry-run.log artifact-sha256.txt; do
    [[ -f "$P4/tiers/$tier/$f" ]] && mv "$P4/tiers/$tier/$f" "$P4/tiers/$tier/k-based/"
  done
  for f in fit-plan.json fit-recipe.json fit-tensor-types.txt fit-oracle-dry-run.log artifact-sha256.txt; do
    [[ -f "$P5/tiers/$tier/$f" ]] && cp "$P5/tiers/$tier/$f" "$P4/tiers/$tier/"
  done
  rm -f "$BUNDLE/$old" "$BUNDLE/$old.quantize-record.json"
  cp "$P5/tiers/$tier/fit-plan.json" "$P5/tiers/$tier/fit-recipe.json" \
     "$P5/tiers/$tier/fit-tensor-types.txt" "$BUNDLE/fit-plans/$tier/"
done < "$P5/tiers.csv"

python3 - <<'EOF'
from pathlib import Path
p4 = Path("experiments/2026-08-29-p4-release-batch/tiers.csv")
new = {
    "FIT-12.5G": "IQ3_M,IQ4_XS,13421772800",
    "FIT-13G": "IQ3_M,IQ4_XS,13958643712",
    "FIT-13.5G": "IQ3_M,IQ4_XS,14495514624",
}
lines = p4.read_text(encoding="utf-8").splitlines()
out = [lines[0]]
for line in lines[1:]:
    tier = line.split(",")[0]
    out.append(f"{tier},{new[tier]}" if tier in new else line)
p4.write_text("\n".join(out) + "\n", encoding="utf-8")
print("P4 tiers.csv: 3 rows moved to the K-free IQ3_M->IQ4_XS pair")
EOF

python3 scripts/summarize_p4.py
cp "$P4/results/comparison-table.md" "$P4/results/kl-curve.png" \
   "$P4/results/sametop-curve.png" "$P4/results/p4-results.json" "$BUNDLE/results/"
python3 scripts/evaluate_p4_gates.py

echo "STAGE P5 DONE: release swapped, results re-rendered, P4 gates re-run"
