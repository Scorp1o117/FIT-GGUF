#!/usr/bin/env bash
# M13 stage B: 6 artifacts x 5 holdout-3 domains = 30 evaluation runs, then summarize.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m13
REFDIR=experiments/2026-08-28-m13-budget-rule/artifacts/reference-logits
M13=experiments/2026-08-28-m13-budget-rule
LOGDIR=$M13/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [orig-fit25]="$FITDIR/Huihui-Qwen3.8-27B-FIT-25.gguf"
  [orig-fit50]="$FITDIR/Huihui-Qwen3.8-27B-FIT-50.gguf"
  [orig-fit75]="$FITDIR/Huihui-Qwen3.8-27B-FIT-75.gguf"
  [v01b-fit25]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT25.gguf"
  [v01b-fit50]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT50.gguf"
  [v01b-fit75]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT75.gguf"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in orig-fit25 orig-fit50 orig-fit75 v01b-fit25 v01b-fit50 v01b-fit75; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d}"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$HOLD/holdout3-${d}-64k.txt" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/holdout3-bf16-${d}.kld" \
      > "$log" 2>&1
    grep "Mean    KLD" "$log" | tail -1
  done
done

python3 scripts/summarize_m13.py "$M13" "$M13/m13-results.json"
echo "STAGE M13-B DONE"
