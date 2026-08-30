#!/usr/bin/env bash
# P6: IQ2 span fix (8G / 8.5G) + Q3_K-free 9.5G / 10G (owner directive).
# analyses -> plans -> retire old artifacts -> quantizes -> evals -> gates
# G1-G6 -> (on pass) release swap -> re-summarize -> P4 gate re-run.
# A gate failure stops BEFORE the swap; retired artifacts stay recoverable.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src

P6=experiments/2026-08-30-p6-iq2-span-fix
P4=experiments/2026-08-29-p4-release-batch
BUNDLE=Qwen3.8-27B-Uncensored-FIT-GGUF
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
IMX=imatrix_unsloth.gguf
RT=tools/llama-b10666-rocm
LOGDIR=$P4/artifacts/logs
REFDIR=$P4/artifacts/reference-kld
EVAL=/run/media/s117/OS/Models/eval-data
MODEL=Qwen3.8-27B-Uncensored
RETired=artifacts/fit/release/retired-p6
mkdir -p "$P6/tiers/_analysis" "$P6/results/baseline" "$LOGDIR" "$RETired"

free_kb=$(df --output=avail /run/media/s117/OS | tail -1)
(( free_kb / 1048576 >= 60 )) || { echo "LOW DISK: ${free_kb}kB free" >&2; exit 1; }

# 0. baseline snapshot of the current release results (before any overwrite)
cp -n "$P4/results/p4-results.json" "$P6/results/baseline/"
cp -n "$P4/results/comparison-table.md" "$P6/results/baseline/"

declare -A SLICES=(
  [wiki_test]="$EVAL/kl-eval-64k.txt"
  [wiki_valid]="$EVAL/kl-eval-valid-64k.txt"
  [chinese]="$EVAL/kl-eval-cn-64k.txt"
  [code]="$EVAL/kl-eval-code-64k.txt"
  [agent_chat]="$EVAL/kl-eval-agent-64k.txt"
)

# Stage A: per-pair analyses (new pairs only), then plans
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  key="${lower}-${upper}"
  if [[ ! -f "$P6/tiers/_analysis/$key/analysis.json" ]]; then
    echo "== analyze pair $key =="
    python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
      --runtime "$RT" --lower "$lower" --upper "$upper" \
      --out-dir "$P6/tiers/_analysis/$key" --skip-hash \
      > "$P6/tiers/analyze-$key.log" 2>&1
  fi
  if [[ ! -f "$P6/tiers/$tier/fit-plan.json" ]]; then
    echo "== plan $tier (target $target) =="
    python3 -m fit_gguf.cli plan --analysis "$P6/tiers/_analysis/$key/analysis.json" \
      --target-bytes "$target" --policy balanced \
      --model-name "$MODEL" --out-prefix "$P6/tiers/$tier/fit" \
      > "$P6/tiers/$tier-plan.log" 2>&1
  fi
  grep -E "dominant|filename|predicted" "$P6/tiers/$tier-plan.log" || tail -3 "$P6/tiers/$tier-plan.log"
done < "$P6/tiers.csv"

# Stage B0: retire old artifacts + stale quantize records BEFORE new quantizes.
# Covers both the old names (baseline snapshot) and the new names (fresh
# plans), in the bundle and in artifacts/fit/release (original P4 records).
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  # Resume guard: never retire a tier whose P6 artifact is already
  # hash-recorded from a prior run.
  [[ -f "$P6/tiers/$tier/artifact-sha256.txt" ]] && continue
  old=$(python3 -c "
import json;print(json.load(open('$P6/results/baseline/p4-results.json'))['artifacts']['$tier']['suggested_filename'])")
  new=$(python3 -c "
import json;print(json.load(open('$P6/tiers/$tier/fit-plan.json'))['suggested_filename'])")
  for name in "$old" "$new"; do
    for base in "$BUNDLE" "artifacts/fit/release"; do
      [[ -f "$base/$name" ]] && mv "$base/$name" "$RETired/"
      [[ -f "$base/$name.quantize-record.json" ]] && mv "$base/$name.quantize-record.json" "$RETired/"
    done
  done
done < "$P6/tiers.csv"
echo "retired old artifacts -> $RETired"

# Stage B: quantizes
quantize_tier () { # tier
  local tier=$1
  local plan="$P6/tiers/$tier/fit-plan.json"
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
      --tensor-types "$P6/tiers/$tier/fit-tensor-types.txt" \
      --out "$out" --expect-bytes "$expect" \
      > "$P6/tiers/$tier-quantize.log" 2>&1
  fi
  sha256sum "$out" | tee "$P6/tiers/$tier/artifact-sha256.txt"
}
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  quantize_tier "$tier"
done < "$P6/tiers.csv"

# Stage C: evals (sentinel guard keyed on artifact sha - P5 amendment 1)
evaluate_tier () { # tier
  local tier=$1
  local plan="$P6/tiers/$tier/fit-plan.json"
  local out="$BUNDLE/$(python3 -c "
import json;print(json.load(open('$plan'))['suggested_filename'])")"
  local sha sent
  sha=$(sha256sum "$out" | cut -d" " -f1)
  for d in wiki_test wiki_valid chinese code agent_chat; do
    local log="$LOGDIR/eval-$tier-$d.log"
    sent="$LOGDIR/eval-$tier-$d.sha"
    if [[ ! -f "$log" || ! -f "$sent" || "$(cat "$sent")" != "$sha" ]]; then
      rm -f "$log"
      echo "== eval $tier $d =="
      "$RT/llama-perplexity" -m "$out" -f "${SLICES[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFDIR/bf16-$d.kld" > "$log" 2>&1
      echo "$sha" > "$sent"
    fi
  done
}
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  evaluate_tier "$tier"
done < "$P6/tiers.csv"

# Gates G1-G6 (exit stops the script before the swap on any failure)
python3 scripts/evaluate_p6_gates.py

# Release swap (only reached with ALL_PASS=True)
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  mkdir -p "$P4/tiers/$tier/prev-span" "$BUNDLE/fit-plans/$tier"
  for f in fit-plan.json fit-recipe.json fit-tensor-types.txt fit-oracle-dry-run.log artifact-sha256.txt; do
    [[ -f "$P4/tiers/$tier/$f" ]] && mv "$P4/tiers/$tier/$f" "$P4/tiers/$tier/prev-span/"
  done
  for f in fit-plan.json fit-recipe.json fit-tensor-types.txt fit-oracle-dry-run.log artifact-sha256.txt; do
    [[ -f "$P6/tiers/$tier/$f" ]] && cp "$P6/tiers/$tier/$f" "$P4/tiers/$tier/"
  done
  cp "$P6/tiers/$tier/fit-plan.json" "$P6/tiers/$tier/fit-recipe.json" \
     "$P6/tiers/$tier/fit-tensor-types.txt" "$BUNDLE/fit-plans/$tier/"
done < "$P6/tiers.csv"

python3 - <<'EOF'
from pathlib import Path
p4 = Path("experiments/2026-08-29-p4-release-batch/tiers.csv")
new = {
    "FIT-8G": "IQ2_XXS,IQ2_M,8589934592",
    "FIT-8.5G": "IQ2_XXS,IQ2_M,9126805504",
    "FIT-9G": "IQ2_XXS,Q2_K_S,9663676416",
    "FIT-9.5G": "IQ2_M,Q2_K_S,10200547328",
    "FIT-10G": "Q2_K_S,IQ3_XXS,10737418240",
}
lines = p4.read_text(encoding="utf-8").splitlines()
out = [lines[0]]
for line in lines[1:]:
    tier = line.split(",")[0]
    out.append(f"{tier},{new[tier]}" if tier in new else line)
p4.write_text("\n".join(out) + "\n", encoding="utf-8")
print("P4 tiers.csv: 4 rows moved to the P6 spans")
EOF

python3 scripts/summarize_p4.py
cp "$P4/results/comparison-table.md" "$P4/results/kl-curve.png" \
   "$P4/results/sametop-curve.png" "$P4/results/p4-results.json" "$BUNDLE/results/"
python3 scripts/evaluate_p4_gates.py

rm -rf "$RETired"
echo "STAGE P6 DONE: release swapped, results re-rendered, P4 gates re-run"
