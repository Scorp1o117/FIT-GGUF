#!/usr/bin/env bash
# M16 stage A+B: holdout-6 Granite BF16 references, then quantize 11 artifacts.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/granite-4.2-8b-BF16.gguf
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m16
M16=experiments/2026-08-29-m16-granite-reveal
REFDIR=$M16/artifacts/reference-logits
LOGDIR=$M16/artifacts/logs
FITDIR=artifacts/fit
IMX=imatrix-granite-apex-c512.gguf
mkdir -p "$REFDIR" "$LOGDIR"

echo "== BF16 SHA-256 check =="
echo "d82690e0dc827f2c43effeb3  (prefix)" ; sha256sum "$SRC" | cut -c1-24

for d in wiki_test wiki_valid chinese code agent_chat; do
  echo "== reference: $d =="
  "$RT/llama-perplexity" \
    -m "$SRC" \
    -f "$HOLD/holdout6-$d-64k.txt" \
    -ngl 99 -t 16 -c 512 -b 512 \
    --kl-divergence-base "$REFDIR/holdout6-bf16-$d.kld" \
    > "$LOGDIR/ref-holdout6-bf16-$d.log" 2>&1
  grep "Final estimate" "$LOGDIR/ref-holdout6-bf16-$d.log"
done

quantize () {  # name, expected_bytes, typefile(optional)
  local name=$1 expect=$2 tf=$3
  local out="$FITDIR/Granite-4.2-8B-${name}.gguf"
  if [[ ! -f "$out" ]]; then
    echo "== quantizing $name =="
    if [[ -n "$tf" ]]; then
      "$RT/llama-quantize" --imatrix "$IMX" --tensor-type-file "$M16/$tf" "$SRC" "$out" IQ3_M > "$LOGDIR/quantize-${name}.log" 2>&1
    else
      "$RT/llama-quantize" --imatrix "$IMX" "$SRC" "$out" "$name" > "$LOGDIR/quantize-${name}.log" 2>&1
    fi
  fi
  local actual
  actual=$(stat -c %s "$out")
  echo "$name actual=$actual expected=$expect"
  [[ "$actual" == "$expect" ]] || { echo "$name SIZE MISMATCH" >&2; exit 1; }
}

quantize IQ3_M 4089184640 ""
quantize IQ4_XS 4820287872 ""
quantize O-FIT25 4271391104 o-fit25-tensor-types.txt
quantize B-FIT25 4271931776 b-fit25-tensor-types.txt
quantize O-FIT50 4454351232 o-fit50-tensor-types.txt
quantize B-FIT50 4454564224 b-fit50-tensor-types.txt
quantize O-FIT75 4637311360 o-fit75-tensor-types.txt
quantize B-FIT75 4637311360 b-fit75-tensor-types.txt
quantize R1-FIT50 4454613376 r1-fit50-tensor-types.txt
quantize R2-FIT50 4454449536 r2-fit50-tensor-types.txt
quantize R3-FIT50 4454629760 r3-fit50-tensor-types.txt
echo "STAGE M16-A+B DONE"
