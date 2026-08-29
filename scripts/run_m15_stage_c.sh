#!/usr/bin/env bash
# M15 stage C: 10 artifacts x 5 holdout-5 domains = 50 evaluation runs.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m15
M15=experiments/2026-08-29-m15-random-baseline
REFDIR=$M15/artifacts/reference-logits
LOGDIR=$M15/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [o25]="$FITDIR/Huihui-Qwen3.8-27B-FIT-25.gguf"
  [b25]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT25.gguf"
  [o75]="$FITDIR/Huihui-Qwen3.8-27B-FIT-75.gguf"
  [b75]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT75.gguf"
  [r1-25]="$FITDIR/Huihui-Qwen3.8-27B-M15-V1-FIT25.gguf"
  [r2-25]="$FITDIR/Huihui-Qwen3.8-27B-M15-V2-FIT25.gguf"
  [r3-25]="$FITDIR/Huihui-Qwen3.8-27B-M15-V3-FIT25.gguf"
  [r1-75]="$FITDIR/Huihui-Qwen3.8-27B-M15-V1-FIT75.gguf"
  [r2-75]="$FITDIR/Huihui-Qwen3.8-27B-M15-V2-FIT75.gguf"
  [r3-75]="$FITDIR/Huihui-Qwen3.8-27B-M15-V3-FIT75.gguf"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in o25 b25 o75 b75 r1-25 r2-25 r3-25 r1-75 r2-75 r3-75; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d}"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$HOLD/holdout5-${d}-64k.txt" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/holdout5-bf16-${d}.kld" \
      > "$log" 2>&1
    grep "Mean    KLD" "$log" | tail -1
  done
done

python3 scripts/summarize_m15.py "$M15" "$M15/m15-results.json"
echo "STAGE M15-C DONE"
