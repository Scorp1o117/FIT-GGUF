#!/usr/bin/env bash
# M11 stage C: 5 variants x 5 holdout domains, fixed protocol, then summarize.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m11
REFDIR=experiments/2026-08-28-m11-holdout/artifacts/reference-logits
LOGDIR=experiments/2026-08-28-m11-holdout/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [fit50]="$FITDIR/Huihui-Qwen3.8-27B-FIT-50.gguf"
  [block-balanced-fit50]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT50.gguf"
  [random-v1]="$FITDIR/Huihui-Qwen3.8-27B-RANDOM-FIT50.gguf"
  [random-v2]="$FITDIR/Huihui-Qwen3.8-27B-RANDOM-FIT50-V2.gguf"
  [random-v3]="$FITDIR/Huihui-Qwen3.8-27B-RANDOM-FIT50-V3.gguf"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in fit50 block-balanced-fit50 random-v1 random-v2 random-v3; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d} (already done)"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$HOLD/holdout-${d}-64k.txt" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/holdout-bf16-${d}.kld" \
      > "$log" 2>&1
    tail -1 "$log"
  done
done

python3 scripts/summarize_m11.py experiments/2026-08-28-m11-holdout experiments/2026-08-28-m11-holdout/holdout-results.json
echo "STAGE C DONE"
