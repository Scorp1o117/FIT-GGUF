#!/usr/bin/env bash
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src
P4=experiments/2026-08-29-p4-release-batch
SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
IMX=imatrix_unsloth.gguf
RT=tools/llama-b10666-rocm
OUTDIR=Qwen3.8-27B-Uncensored-FIT-GGUF
LOGDIR=$P4/artifacts/logs

tail -3 "$P4/tiers.csv" > "$P4/tiers-ext.csv"
while IFS=, read -r tier lower upper target; do
  key="${lower}-${upper}"
  if [[ ! -f "$P4/tiers/_analysis/$key/analysis.json" ]]; then
    echo "== analyze pair $key =="
    python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
      --runtime "$RT" --lower "$lower" --upper "$upper" \
      --out-dir "$P4/tiers/_analysis/$key" --skip-hash
  fi
  if [[ ! -f "$P4/tiers/$tier/fit-plan.json" ]]; then
    echo "== plan $tier =="
    python3 -m fit_gguf.cli plan --analysis "$P4/tiers/_analysis/$key/analysis.json" \
      --target-bytes "$target" --policy balanced --model-name "Qwen3.8-27B-Uncensored" \
      --out-prefix "$P4/tiers/$tier/fit"
  fi
  plan="$P4/tiers/$tier/fit-plan.json"
  out="$OUTDIR/$(python3 -c "import json;print(json.load(open('$plan'))['suggested_filename'])")"
  expect=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['predicted_size_bytes'])" "$plan")
  analysis=$(python3 -c "import json;print(json.load(open('$plan'))['analysis_path'])")
  if [[ ! -f "$out" ]]; then
    echo "== quantize $tier (expect $expect) =="
    python3 -m fit_gguf.cli quantize --analysis "$analysis" \
      --tensor-types "$P4/tiers/$tier/fit-tensor-types.txt" \
      --out "$out" --expect-bytes "$expect"
  fi
  sha256sum "$out" | tee "$P4/tiers/$tier/artifact-sha256.txt"
  mkdir -p "$OUTDIR/fit-plans/$tier"
  for s in fit-plan.json fit-recipe.json fit-tensor-types.txt; do
    cp "$P4/tiers/$tier/$s" "$OUTDIR/fit-plans/$tier/$s"
  done
done < "$P4/tiers-ext.csv"

REFDIR=$P4/artifacts/reference-kld
EVAL=/run/media/s117/OS/Models/eval-data
declare -A SLICES=(
  [wiki_test]="$EVAL/kl-eval-64k.txt" [wiki_valid]="$EVAL/kl-eval-valid-64k.txt"
  [chinese]="$EVAL/kl-eval-cn-64k.txt" [code]="$EVAL/kl-eval-code-64k.txt"
  [agent_chat]="$EVAL/kl-eval-agent-64k.txt"
)
while IFS=, read -r tier lower upper target; do
  out="$OUTDIR/$(python3 -c "import json;print(json.load(open('$P4/tiers/$tier/fit-plan.json'))['suggested_filename'])")"
  for d in wiki_test wiki_valid chinese code agent_chat; do
    log="$LOGDIR/eval-$tier-$d.log"
    if ! grep -q "Mean.*KLD" "$log" 2>/dev/null; then
      echo "== eval $tier $d =="
      "$RT/llama-perplexity" -m "$out" -f "${SLICES[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFDIR/bf16-$d.kld" > "$log" 2>&1
    fi
  done
done < "$P4/tiers-ext.csv"
echo "STAGE P4-EXT DONE"
