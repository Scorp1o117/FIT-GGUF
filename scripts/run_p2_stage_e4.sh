#!/usr/bin/env bash
# P2 stage E4: three preregistered probe plans + quantizes, zero-byte gates.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src

P2=experiments/2026-08-29-p2-full-envelope
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
IMX=imatrix_unsloth.gguf
RT=tools/llama-b10666-rocm
FITDIR=artifacts/fit/p2-probes
mkdir -p "$P2/probes" "$FITDIR"

probe () { # lower upper name
  local lower=$1 upper=$2 name=$3
  echo "== probe $name: analyze ($lower -> $upper) =="
  python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
    --runtime "$RT" --lower "$lower" --upper "$upper" \
    --out-dir "$P2/probes/$name" --skip-hash
  echo "== probe $name: plan =="
  python3 -m fit_gguf.cli plan --analysis "$P2/probes/$name/analysis.json" \
    --fit 0.5 --policy original --out-prefix "$P2/probes/$name/fit50"
  local expect
  expect=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['predicted_size_bytes'])" "$P2/probes/$name/fit50-plan.json")
  echo "== probe $name: quantize (expect $expect) =="
  python3 -m fit_gguf.cli quantize --analysis "$P2/probes/$name/analysis.json" \
    --tensor-types "$P2/probes/$name/fit50-tensor-types.txt" \
    --out "$FITDIR/${name}-FIT50.gguf" --expect-bytes "$expect"
  sha256sum "$FITDIR/${name}-FIT50.gguf" | tee "$P2/probes/$name/artifact-sha256.txt"
}

probe IQ1_M IQ3_XXS probe-low
probe Q5_K_S Q5_K_M probe-mid
probe Q6_K Q8_0 probe-top

echo "STAGE P2-E4 DONE"
