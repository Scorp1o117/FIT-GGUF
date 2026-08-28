#!/usr/bin/env bash
# M14 stage C: 9 artifacts x 5 holdout-4 domains = 45 evaluation runs, then summarize.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m14
M14=experiments/2026-08-28-m14-swap-ablation
REFDIR=$M14/artifacts/reference-logits
LOGDIR=$M14/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [orig-fit50]="$FITDIR/Huihui-Qwen3.8-27B-FIT-50.gguf"
  [v01b-fit50]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT50.gguf"
  [oe-fit50]="$FITDIR/Huihui-Qwen3.8-27B-M14-OE-FIT50.gguf"
  [bl-fit50]="$FITDIR/Huihui-Qwen3.8-27B-M14-BL-FIT50.gguf"
  [shuf-fit50]="$FITDIR/Huihui-Qwen3.8-27B-M14-SHUF-FIT50.gguf"
  [orig-fit75]="$FITDIR/Huihui-Qwen3.8-27B-FIT-75.gguf"
  [v01b-fit75]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT75.gguf"
  [oe-fit75]="$FITDIR/Huihui-Qwen3.8-27B-M14-OE-FIT75.gguf"
  [bl-fit75]="$FITDIR/Huihui-Qwen3.8-27B-M14-BL-FIT75.gguf"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in orig-fit50 v01b-fit50 oe-fit50 bl-fit50 shuf-fit50 orig-fit75 v01b-fit75 oe-fit75 bl-fit75; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d}"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$HOLD/holdout4-${d}-64k.txt" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/holdout4-bf16-${d}.kld" \
      > "$log" 2>&1
    grep "Mean    KLD" "$log" | tail -1
  done
done

python3 scripts/summarize_m14.py "$M14" "$M14/m14-results.json"
echo "STAGE M14-C DONE"
