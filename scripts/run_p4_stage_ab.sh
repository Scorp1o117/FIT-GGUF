#!/usr/bin/env bash
# P4 stage A+B: per-pair analyses, 11 balanced plans, 11 FIT quantizes,
# 14 preset reference quantizes. Skip-if-done guards on every artifact.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src

P4=experiments/2026-08-29-p4-release-batch
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
IMX=imatrix_unsloth.gguf
RT=tools/llama-b10666-rocm
OUTDIR=artifacts/fit/release
mkdir -p "$P4/tiers" "$P4/tiers/_analysis" "$OUTDIR/refs" "$P4/artifacts/logs"

expect_ladder () { python3 -c "
import json,sys
d=json.load(open('experiments/2026-08-29-p2-full-envelope/preset-ladder.json'))
print(d['ladder'][sys.argv[1]]['predicted_size_bytes'])" "$1"; }

plan_tier () { # tier lower upper target
  local tier=$1 lower=$2 upper=$3 target=$4
  local key="${lower}-${upper}"
  if [[ ! -f "$P4/tiers/_analysis/$key/analysis.json" ]]; then
    echo "== analyze pair $key =="
    python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
      --runtime "$RT" --lower "$lower" --upper "$upper" \
      --out-dir "$P4/tiers/_analysis/$key" --skip-hash
  fi
  if [[ ! -f "$P4/tiers/$tier/fit-plan.json" ]]; then
    echo "== plan $tier =="
    python3 -m fit_gguf.cli plan --analysis "$P4/tiers/_analysis/$key/analysis.json" \
      --target-bytes "$target" --policy balanced --out-prefix "$P4/tiers/$tier/fit"
  fi
}

quantize_tier () { # tier
  local tier=$1
  local plan="$P4/tiers/$tier/fit-plan.json"
  local out="$OUTDIR/$(python3 -c "
import json;print(json.load(open('$plan'))['suggested_filename'])")"
  local expect analysis
  expect=$(python3 -c "
import json,sys;print(json.load(open(sys.argv[1]))['predicted_size_bytes'])" "$plan")
  analysis=$(python3 -c "
import json;print(json.load(open('$plan'))['analysis_path'])")
  if [[ ! -f "$out" ]]; then
    echo "== quantize $tier -> $out (expect $expect) =="
    python3 -m fit_gguf.cli quantize --analysis "$analysis" \
      --tensor-types "$P4/tiers/$tier/fit-tensor-types.txt" \
      --out "$out" --expect-bytes "$expect"
  fi
  sha256sum "$out" | tee "$P4/tiers/$tier/artifact-sha256.txt"
}

# Stage A: analyses + plans
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  plan_tier "$tier" "$lower" "$upper" "$target"
done < "$P4/tiers.csv"

# Stage B1: FIT quantizes
while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  quantize_tier "$tier"
done < "$P4/tiers.csv"

# Stage B2: preset references
while read -r preset; do
  [[ "$preset" == "preset" ]] && continue
  local_out="$OUTDIR/refs/$preset.gguf"
  expect=$(expect_ladder "$preset")
  if [[ ! -f "$local_out" ]]; then
    echo "== reference $preset (expect $expect) =="
    "$RT/llama-quantize" --imatrix "$IMX" "$SRC" "$local_out" "$preset" \
      > "$P4/artifacts/logs/quantize-ref-$preset.log" 2>&1
  fi
  actual=$(stat -c %s "$local_out")
  [[ "$actual" == "$expect" ]] || { echo "REF $preset SIZE MISMATCH: $actual != $expect" >&2; exit 1; }
  sha256sum "$local_out" | tee "$P4/refs/$preset-sha256.txt"
done < "$P4/refs.csv"

echo "STAGE P4-A+B DONE"
