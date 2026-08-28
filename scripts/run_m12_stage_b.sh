#!/usr/bin/env bash
# M12 stage B: quantize block-balanced FIT-25 and FIT-75 from the generated
# tensor-type files, then verify exact predicted sizes.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
M12=experiments/2026-08-28-m12-block-balanced-curve
LOGDIR=$M12/artifacts/logs
mkdir -p "$LOGDIR"

quantize () {  # name, expected_bytes
  local name=$1 expect=$2
  local out="artifacts/fit/Huihui-Qwen3.8-27B-BLOCK-BALANCED-${name}.gguf"
  if [[ ! -f "$out" ]]; then
    echo "== quantizing $name =="
    "$RT/llama-quantize" \
      --imatrix imatrix_unsloth.gguf \
      --tensor-type-file "$M12/block-balanced-${name,,}-tensor-types.txt" \
      "$SRC" "$out" IQ3_M \
      > "$LOGDIR/quantize-${name}.log" 2>&1
  fi
  local actual
  actual=$(stat -c %s "$out")
  echo "$name actual=$actual expected=$expect"
  [[ "$actual" == "$expect" ]] || { echo "$name SIZE MISMATCH" >&2; exit 1; }
}

quantize FIT25 13205208032
quantize FIT75 14454856672
echo "STAGE M12-B DONE"
