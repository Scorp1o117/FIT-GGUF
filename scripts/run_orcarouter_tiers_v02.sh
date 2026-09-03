#!/usr/bin/env bash
# orcarouter four-tier FIT v0.2 build (goal deliverable, GPT design pending):
# one artifact per guard tier at the calibration-crossing size, built with the
# v0.2 stack = Refine C_role re-weighting + G2 re-finalization gate + eval-v1.
# Preset and FIT v0.1 columns reuse existing evaluated artifacts.
# Hot I/O: source BF16 (53.8G) read from NTFS read-only; artifacts on tmpfs,
# deleted after eval. NVMe is never written.
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
SRC=/run/media/s117/OS/Models/orcarouter-Qwen3.8-27B-Uncensored/Qwen3.8-27B-Uncensored-BF16.gguf
IMX=/run/media/s117/OS/FIT-GGUF/imatrix_unsloth.gguf
REFS=$M2/../2026-09-02-eval-v1/refs-orcarouter
QROOT=/dev/shm/orca-v2
LOG=$QROOT/logs
ART=$QROOT/artifacts
mkdir -p "$LOG" "$ART"
PROFILE=profiles/refine-profile-qwen-hybrid-dev1.json

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/v2-progress.log"; }

declare -A PAIR=(
  [Quality]="Q4_K_M Q5_K_M"
  [Balanced]="IQ3_M IQ4_XS"
  [Compact]="Q3_K_S IQ3_S"
  [Mini]="Q2_K IQ3_XXS"
)
declare -A TARGET=(
  [Quality]=17857895277
  [Balanced]=13901839675
  [Compact]=12259844742
  [Mini]=11117618267
)

for TIER in Quality Balanced Compact Mini; do
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$M2/logs/eval-orcorouter-FIT-V2-$TIER-$d.log" ]] && grep -q "Mean.*KLD" "$M2/logs/eval-orcorouter-FIT-V2-$TIER-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip FIT-V2-$TIER (done)"; continue; fi

  read -r LOWER UPPER <<< "${PAIR[$TIER]}"
  ANDIR="$M2/orcarouter-analysis-$LOWER-$UPPER"
  if [[ ! -s "$ANDIR/analysis.json" ]]; then
    say "fit analyze $LOWER->$UPPER"
    PYTHONPATH=src python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
      --runtime "$RT" --out-dir "$ANDIR" --lower "$LOWER" --upper "$UPPER" \
      --imatrix-arg "$IMX" --skip-hash > "$LOG/fit-analyze-$LOWER-$UPPER.log" 2>&1 \
      || { say "fit analyze FAILED $LOWER-$UPPER"; continue; }
  fi

  PREFIX="$M2/orcarouter-fit-v2-$TIER"
  if [[ ! -s "${PREFIX}-plan.json" ]]; then
    say "fit plan FIT-V2-$TIER target=${TARGET[$TIER]} (C_role on)"
    PYTHONPATH=src python3 -m fit_gguf.cli plan --analysis "$ANDIR/analysis.json" \
      --target-bytes "${TARGET[$TIER]}" --policy balanced \
      --refine-profile "$PROFILE" --model-name "orcarouter-Qwen3.8-27B-Uncensored-FIT-V2-$TIER" \
      --out-prefix "$PREFIX" > "$LOG/fit-plan-$TIER.log" 2>&1 \
      || { say "fit plan FAILED FIT-V2-$TIER"; continue; }
    say "FIT-V2-$TIER planned: $(python3 -c "import json;r=json.load(open('${PREFIX}-plan.json'));print(r['predicted_size_bytes'], r['dominant_qtype'])")"
  fi

  tmpout=$ART/orcarouter-FIT-V2-$TIER.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "fit quantize FIT-V2-$TIER (G2 gate)"
    PYTHONPATH=src python3 -m fit_gguf.cli quantize --analysis "$ANDIR/analysis.json" \
      --tensor-types "${PREFIX}-tensor-types.txt" --out "$tmpout" \
      --imatrix-arg "$IMX" > "$LOG/fit-quantize-$TIER.log" 2>&1 \
      || { say "fit quantize FAILED FIT-V2-$TIER"; continue; }
    say "FIT-V2-$TIER $(grep 'G2:' "$LOG/fit-quantize-$TIER.log" | tail -1)"
  fi

  for d in "${DOMAINS[@]}"; do
    log="$M2/logs/eval-orcorouter-FIT-V2-$TIER-$d.log"
    [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" && continue
    for attempt in 1 2 3; do
      rm -f "$log"
      "$RT/llama-perplexity" -m "$tmpout" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
      grep -q "Mean.*KLD" "$log" && break
      say "eval FIT-V2-$TIER/$d attempt $attempt failed; backing off"
      sleep 15
    done
    grep -q "Mean.*KLD" "$log" && say "eval FIT-V2-$TIER/$d OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')" \
      || say "eval FIT-V2-$TIER/$d FAILED"
    sleep 5
  done

  eval_ok=true
  for d in "${DOMAINS[@]}"; do
    log="$M2/logs/eval-orcorouter-FIT-V2-$TIER-$d.log"
    [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" || eval_ok=false
  done
  if [[ "$eval_ok" == true ]]; then
    if ! grep -q "^orcorouter-FIT-V2-$TIER  " "$M2/results/artifact-manifest.txt" 2>/dev/null; then
      printf 'orcorouter-FIT-V2-%s  %s  %s\n' "$TIER" "$(stat -c %s "$tmpout")" "$(sha256sum "$tmpout" | cut -d' ' -f1)" >> "$M2/results/artifact-manifest.txt"
    fi
    rm -f "$tmpout"
    say "FIT-V2-$TIER hash-recorded and released (tmpfs)"
  else
    say "FIT-V2-$TIER evals incomplete — artifact kept for resume"
  fi
  sync
done

say "================ ORCAROUTER FIT-V2 TIERS DONE ================"
