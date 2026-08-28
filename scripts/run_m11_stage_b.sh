#!/usr/bin/env bash
# M11 stage B: rebuild the three deleted random FIT-50 GGUFs from their retained
# tensor-type files and verify the rebuilt SHA-256 against the M10 record.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
M10=experiments/2026-08-28-m10-ablation
FITDIR=artifacts/fit
LOGDIR=experiments/2026-08-28-m11-holdout/artifacts/logs
mkdir -p "$LOGDIR"

rebuild () {  # name, expected_sha, tensor_type_file
  local name=$1 expect=$2 ttfile=$3
  local out="$FITDIR/Huihui-Qwen3.8-27B-${name}.gguf"
  if [[ -f "$out" ]]; then
    echo "== $name already present, verifying =="
  else
    echo "== rebuilding $name =="
    # llama-quantize parses flags only before the positional args (model, output, type).
    "$RT/llama-quantize" \
      --imatrix imatrix_unsloth.gguf \
      --tensor-type-file "$M10/$ttfile" \
      "$SRC" "$out" IQ3_M \
      > "$LOGDIR/rebuild-${name}.log" 2>&1
  fi
  local actual
  actual=$(sha256sum "$out" | cut -d' ' -f1)
  if [[ "$actual" == "$expect" ]]; then
    echo "$name SHA-256 OK: $actual"
  else
    echo "$name SHA-256 MISMATCH: got $actual expected $expect" >&2
    exit 1
  fi
}

rebuild RANDOM-FIT50    dba2ec218beeed0cc47fc54bbc76c9ae97d4e24298931d940da29b7365f94892 random-fit50-tensor-types.txt
rebuild RANDOM-FIT50-V2 2fd1a30b1b671cc223085d920d88f3239b9580d7076852ade74ebab9fcfcfeb4 random-fit50-v2-tensor-types.txt
rebuild RANDOM-FIT50-V3 c0a119b8961bdec5402f41ee29fede3659e2005c90dc02434191dbafbd7dce40 random-fit50-v3-tensor-types.txt

echo "== retained variant hashes =="
echo "e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306  $FITDIR/Huihui-Qwen3.8-27B-FIT-50.gguf" | sha256sum -c -
echo "7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07  $FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT50.gguf" | sha256sum -c -
echo "STAGE B DONE"
