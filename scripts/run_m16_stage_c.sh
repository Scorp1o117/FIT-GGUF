#!/usr/bin/env bash
# M16 stage C: 11 artifacts x 5 holdout-6 domains = 55 evaluation runs.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m16
M16=experiments/2026-08-29-m16-granite-reveal
REFDIR=$M16/artifacts/reference-logits
LOGDIR=$M16/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [iq3m]="$FITDIR/Granite-4.2-8B-IQ3_M.gguf"
  [iq4xs]="$FITDIR/Granite-4.2-8B-IQ4_XS.gguf"
  [o25]="$FITDIR/Granite-4.2-8B-O-FIT25.gguf"
  [b25]="$FITDIR/Granite-4.2-8B-B-FIT25.gguf"
  [o50]="$FITDIR/Granite-4.2-8B-O-FIT50.gguf"
  [b50]="$FITDIR/Granite-4.2-8B-B-FIT50.gguf"
  [o75]="$FITDIR/Granite-4.2-8B-O-FIT75.gguf"
  [b75]="$FITDIR/Granite-4.2-8B-B-FIT75.gguf"
  [r1-50]="$FITDIR/Granite-4.2-8B-R1-FIT50.gguf"
  [r2-50]="$FITDIR/Granite-4.2-8B-R2-FIT50.gguf"
  [r3-50]="$FITDIR/Granite-4.2-8B-R3-FIT50.gguf"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in iq3m o25 b25 o50 b50 o75 b75 iq4xs r1-50 r2-50 r3-50; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d}"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$HOLD/holdout6-${d}-64k.txt" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/holdout6-bf16-${d}.kld" \
      > "$log" 2>&1
    grep "Mean    KLD" "$log" | tail -1
  done
done

python3 scripts/summarize_m16.py "$M16" "$M16/m16-results.json"
echo "STAGE M16-C DONE"
