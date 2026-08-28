#!/usr/bin/env bash
# M12 stage C: evaluate block-balanced FIT-25/FIT-75 against the M9 reference
# logits on the original five M9 slices, then summarize.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
EVAL=/run/media/s117/OS/Models/eval-data
REFDIR=experiments/2026-08-28-m9-fit-curve/artifacts/reference-logits
M12=experiments/2026-08-28-m12-block-balanced-curve
LOGDIR=$M12/artifacts/logs
FITDIR=artifacts/fit

declare -A VARIANT=(
  [block-balanced-fit25]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT25.gguf"
  [block-balanced-fit75]="$FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT75.gguf"
)
declare -A SLICE=(
  [wiki_test]="kl-eval-64k.txt"
  [wiki_valid]="kl-eval-valid-64k.txt"
  [chinese]="kl-eval-cn-64k.txt"
  [code]="kl-eval-code-64k.txt"
  [agent_chat]="kl-eval-agent-64k.txt"
)
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)

for v in block-balanced-fit25 block-balanced-fit75; do
  for d in "${DOMAINS[@]}"; do
    log="$LOGDIR/eval-${v}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then
      echo "skip ${v}/${d} (already done)"
      continue
    fi
    echo "== eval ${v} / ${d} =="
    "$RT/llama-perplexity" \
      -m "${VARIANT[$v]}" \
      -f "$EVAL/${SLICE[$d]}" \
      -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence \
      --kl-divergence-base "$REFDIR/current-bf16-${d}.kld" \
      > "$log" 2>&1
    grep "Mean    KLD" "$log" | tail -1
  done
done
echo "STAGE M12-C DONE"
