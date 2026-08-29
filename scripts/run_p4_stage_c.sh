#!/usr/bin/env bash
# P4 stage C: BF16 KLD references, then KL evals for 11 FIT tiers + 14 presets.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
P4=experiments/2026-08-29-p4-release-batch
EVAL=/run/media/s117/OS/Models/eval-data
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
REFDIR=$P4/artifacts/reference-kld
LOGDIR=$P4/artifacts/logs
OUTDIR=artifacts/fit/release
mkdir -p "$REFDIR" "$LOGDIR"

declare -A SLICES=(
  [wiki_test]="$EVAL/kl-eval-64k.txt"
  [wiki_valid]="$EVAL/kl-eval-valid-64k.txt"
  [chinese]="$EVAL/kl-eval-cn-64k.txt"
  [code]="$EVAL/kl-eval-code-64k.txt"
  [agent_chat]="$EVAL/kl-eval-agent-64k.txt"
)
sha256sum "${SLICES[wiki_test]}" "${SLICES[wiki_valid]}" "${SLICES[chinese]}" \
  "${SLICES[code]}" "${SLICES[agent_chat]}" > "$P4/slices-sha256.txt"

for d in wiki_test wiki_valid chinese code agent_chat; do
  kld="$REFDIR/bf16-$d.kld"
  if [[ ! -f "$kld" ]]; then
    echo "== BF16 reference $d =="
    "$RT/llama-perplexity" -m "$SRC" -f "${SLICES[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence-base "$kld" > "$LOGDIR/ref-bf16-$d.log" 2>&1
  fi
  grep "Final estimate" "$LOGDIR/ref-bf16-$d.log" || true
done

evaluate () { # artifact_path label
  local art=$1 label=$2
  for d in wiki_test wiki_valid chinese code agent_chat; do
    local log="$LOGDIR/eval-$label-$d.log"
    if ! grep -q "Mean.*KLD" "$log" 2>/dev/null; then
      echo "== eval $label $d =="
      "$RT/llama-perplexity" -m "$art" -f "${SLICES[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFDIR/bf16-$d.kld" > "$log" 2>&1
    fi
  done
}

while IFS=, read -r tier lower upper target; do
  [[ "$tier" == "tier" ]] && continue
  out="$OUTDIR/$(python3 -c "import json;print(json.load(open('$P4/tiers/$tier/fit-plan.json'))['suggested_filename'])")"
  evaluate "$out" "$tier"
done < "$P4/tiers.csv"

while read -r preset; do
  [[ "$preset" == "preset" ]] && continue
  evaluate "$OUTDIR/refs/$preset.gguf" "ref-$preset"
done < "$P4/refs.csv"

echo "STAGE P4-C DONE"
