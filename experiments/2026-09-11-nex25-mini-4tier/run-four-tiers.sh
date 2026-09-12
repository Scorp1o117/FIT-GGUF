#!/bin/bash
# Run the four FIT fidelity tiers sequentially for Nex-N2.5-mini-abliterix-v8-t17.
#
# Hard constraints honoured here:
#   * --logs-dir and --work-dir live on ext4. /run/media/s117/OS is ntfs3 and the
#     project documents a kernel BUG (iomap_write_end) when llama.cpp's unbuffered
#     stderr is redirected onto it. Data files on ntfs3 are fine; subprocess logs
#     are not.
#   * one tier at a time, with a free-space gate before each, because a candidate
#     artifact for this model is 12-40 GB and / has a fixed budget.
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

BUNDLE=experiments/2026-09-11-nex25-mini-4tier/calibration
SRC=artifacts/source/Nex-N2.5-mini-abliterix-v8-t17-BF16.gguf
LOGROOT=/home/s117/fit-logs/nex25-fs
SCRATCHROOT=/home/s117/fit-scratch
MIN_FREE_GB=60
STATUS=$LOGROOT/status.txt

mkdir -p "$LOGROOT"
: > "$STATUS"

# The calibration bundle publishes its imatrix alongside the records; fall back
# to the scratch copy if the bundle kept it private.
IMATRIX=""
for cand in "$BUNDLE/calibration-imatrix.gguf" "$BUNDLE/imatrix.gguf" \
            "$BUNDLE/work/imatrix.gguf" \
            "$SCRATCHROOT/nex25-calibrate/calibration-imatrix.gguf"; do
  [ -f "$cand" ] && IMATRIX="$cand" && break
done
[ -n "$IMATRIX" ] || { echo "NO_IMATRIX" >> "$STATUS"; exit 1; }
echo "imatrix=$IMATRIX" >> "$STATUS"

MANIFEST=$(ls "$BUNDLE"/state-artifact-manifest.txt 2>/dev/null | head -1)
if [ -z "$MANIFEST" ]; then
  MANIFEST=$(find "$SCRATCHROOT/nex25-calibrate" -name state-artifact-manifest.txt 2>/dev/null | head -1)
fi
echo "manifest=${MANIFEST:-none}" >> "$STATUS"

REFS=""
for cand in "$BUNDLE/references" "$SCRATCHROOT/nex25-calibrate/references"; do
  [ -d "$cand" ] && REFS="$cand" && break
done
[ -n "$REFS" ] || { echo "NO_REFS" >> "$STATUS"; exit 1; }
echo "refs=$REFS" >> "$STATUS"

declare -A LADDER=(
  [quality]="Q5_K_M,Q6_K"
  [balanced]="IQ3_M,Q6_K"
  [compact]="IQ2_M,Q6_K"
  [mini]="IQ2_XXS,Q4_K_M"
)
FAILED=0
for tier in quality balanced compact mini; do
  FREE=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if [ "$FREE" -lt "$MIN_FREE_GB" ]; then
    echo "ABORT low disk before $tier: ${FREE}G free" >> "$STATUS"
    exit 1
  fi
  if grep -qs 'verified_pass' "experiments/2026-09-11-nex25-mini-4tier/fs/$tier/fidelity-search-$tier-summary.json" 2>/dev/null; then
    echo "SKIP $tier (already verified_pass)" >> "$STATUS"; continue
  fi
  echo "START $tier $(date '+%F %T') free=${FREE}G" >> "$STATUS"
  rm -rf "$SCRATCHROOT/fs-$tier"
  mkdir -p "$SCRATCHROOT/fs-$tier" "$LOGROOT/$tier"
  PYTHONPATH=/run/media/s117/OS/FIT-GGUF/src \
  /home/s117/heretic-env/bin/python -c "
import sys
from fit_gguf.cli import main
sys.argv = ['fit','fidelity-search',
  '--source','$SRC',
  '--imatrix','$IMATRIX',
  '--runtime','tools/llama-b10666-rocm',
  '--refs-dir','$REFS',
  '--eval-data-dir','eval-data',
  '--tier','$tier',
  '--preset-ladder','${LADDER[$tier]}',
  '--manifest','$MANIFEST',
  '--logs-dir','$LOGROOT/$tier',
  '--out-dir','experiments/2026-09-11-nex25-mini-4tier/fs/$tier',
  '--work-dir','$SCRATCHROOT/fs-$tier',
  '--threads','16',
  '--n-gpu-layers','30']
main()
" >> "$LOGROOT/$tier/driver.log" 2>&1
  rc=$?
  echo "END $tier rc=$rc $(date '+%F %T')" >> "$STATUS"
  if [ $rc -ne 0 ]; then
    echo "FAIL $tier rc=$rc (continuing with remaining tiers)" >> "$STATUS"
    FAILED=$((FAILED+1))
  fi
  # keep the scratch small between tiers
  rm -rf "$SCRATCHROOT/fs-$tier"
done
if [ "$FAILED" -eq 0 ]; then echo "ALL_DONE $(date '+%F %T')" >> "$STATUS"; else echo "DONE_WITH_FAILURES=$FAILED $(date '+%F %T')" >> "$STATUS"; fi
