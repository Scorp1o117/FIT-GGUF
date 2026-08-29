#!/usr/bin/env bash
# P2 stage E1+E2: record BF16 SHA-256, then build the APEX c512 imatrix.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
P2=experiments/2026-08-29-p2-full-envelope
CAL=/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt
IMX=imatrix-apex-qwen38-c512.gguf
LOGDIR=$P2/logs
mkdir -p "$LOGDIR"

echo "== E1: BF16 SHA-256 =="
sha256sum "$SRC" | tee "$P2/bf16-sha256.txt"
stat -c "size=%s" "$SRC" | tee -a "$P2/bf16-sha256.txt"

echo "== E2: imatrix (APEX-imatrix-Small.txt, ctx 512, all chunks, ngl 99) =="
"$RT/llama-imatrix" \
  -m "$SRC" \
  -f "$CAL" \
  -c 512 \
  -ngl 99 -t 16 \
  -o "$IMX" \
  > "$LOGDIR/imatrix.log" 2>&1

echo "== E2 done: imatrix KVs =="
sha256sum "$IMX" | tee "$P2/imatrix-sha256.txt"
echo "STAGE P2-E1E2 DONE"
