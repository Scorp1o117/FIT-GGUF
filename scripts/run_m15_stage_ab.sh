#!/usr/bin/env bash
# M15 stage A+B: holdout-5 BF16 references, then quantize six paired random artifacts.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m15
M15=experiments/2026-08-29-m15-random-baseline
REFDIR=$M15/artifacts/reference-logits
LOGDIR=$M15/artifacts/logs
FITDIR=artifacts/fit
mkdir -p "$REFDIR" "$LOGDIR"

echo "== BF16 SHA-256 check =="
echo "8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc  $SRC" | sha256sum -c -

for d in wiki_test wiki_valid chinese code agent_chat; do
  echo "== reference: $d =="
  "$RT/llama-perplexity" \
    -m "$SRC" \
    -f "$HOLD/holdout5-$d-64k.txt" \
    -ngl 99 -t 16 -c 512 -b 512 \
    --kl-divergence-base "$REFDIR/holdout5-bf16-$d.kld" \
    > "$LOGDIR/ref-holdout5-bf16-$d.log" 2>&1
  tail -2 "$LOGDIR/ref-holdout5-bf16-$d.log"
done

quantize () {  # name, expected_bytes
  local name=$1 expect=$2
  local out="$FITDIR/Huihui-Qwen3.8-27B-${name}.gguf"
  if [[ ! -f "$out" ]]; then
    echo "== quantizing $name =="
    "$RT/llama-quantize" \
      --imatrix imatrix_unsloth.gguf \
      --tensor-type-file "$M15/${name,,}-tensor-types.txt" \
      "$SRC" "$out" IQ3_M \
      > "$LOGDIR/quantize-${name}.log" 2>&1
  fi
  local actual
  actual=$(stat -c %s "$out")
  echo "$name actual=$actual expected=$expect"
  [[ "$actual" == "$expect" ]] || { echo "$name SIZE MISMATCH" >&2; exit 1; }
}

quantize M15-V1-FIT25 13206265952
quantize M15-V2-FIT25 13206274912
quantize M15-V3-FIT25 13206258912
quantize M15-V1-FIT75 14456884192
quantize M15-V2-FIT75 14457087072
quantize M15-V3-FIT75 14457094752
echo "STAGE M15-A+B DONE"
