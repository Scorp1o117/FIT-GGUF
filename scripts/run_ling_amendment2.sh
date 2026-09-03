#!/usr/bin/env bash
# M2 amendment-2: Ling window-fill FIT points (6 pinned plans), tmpfs hot loop.
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
QROOT=/dev/shm/m2-ling
LOG=$QROOT/logs
REFS=$QROOT/refs
ART=$QROOT/artifacts
STATE=$QROOT/state
DISK_LOG=$M2/logs-ling
DISK_STATE=$M2/results-ling
MODELDIR="/run/media/s117/OS/Models/Ling-3.0-tiny-abliterated-APEX-GGUF"
mkdir -p "$LOG" "$ART" "$STATE" "$DISK_LOG" "$DISK_STATE"
BF16=$QROOT/Ling-bf16.gguf
IMX=$QROOT/ling-imatrix.dat

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/ling-progress.log"; }

[[ -f "$BF16" ]] || { say "tmpfs BF16 missing"; exit 1; }
for d in "${DOMAINS[@]}"; do [[ -s "$REFS/bf16-$d.kld" ]] || { say "tmpfs ref $d missing"; exit 1; }; done

record() {
  local name=$1 path=$2
  if grep -q "^$name  " "$STATE/artifact-manifest.txt" 2>/dev/null; then return 0; fi
  [[ -f "$path" ]] && printf '%s  %s  %s\n' "$name" "$(stat -c %s "$path")" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$STATE/artifact-manifest.txt" \
    || printf '%s  MISSING\n' "$name" >> "$STATE/artifact-manifest.txt"
}

sync_state() {
  sync
  cp -f "$LOG"/ling-progress.log "$LOG"/eval-*.log "$LOG"/quantize-*.log "$DISK_LOG/" 2>/dev/null
  cp -f "$STATE/artifact-manifest.txt" "$DISK_STATE/" 2>/dev/null
  sync
  if [[ -d /run/media/s117/TF && -w /run/media/s117/TF ]]; then
    mkdir -p /run/media/s117/TF/fit-m2-ling-records
    cp -f "$LOG"/ling-progress.log "$LOG"/eval-*.log "$STATE/artifact-manifest.txt" /run/media/s117/TF/fit-m2-ling-records/ 2>/dev/null
    say "records also copied to TF card"
  fi
}

eval_model() {
  local gguf=$1 tag=$2 d log
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then say "skip eval ${tag}/${d} (done)"; continue; fi
    say "eval ${tag}/${d} start"
    "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
    grep -q "Mean.*KLD" "$log" && say "eval ${tag}/${d} OK" || say "eval ${tag}/${d} FAILED"
  done
}

say "================ amendment-2: Ling window fill ================"
n=0
for spec in "FIT-LA1 L-A1 IQ4_XS" "FIT-LA2 L-A2 IQ4_XS" "FIT-LA3 L-A3 IQ4_XS" \
            "FIT-LB1 L-B1 IQ3_M" "FIT-LB2 L-B2 IQ3_M" "FIT-LB3 L-B3 IQ3_M"; do
  set -- $spec
  TAG=$1; PLAN=$2; LOW=$3
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-ling-$TAG-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-ling-$TAG-$d.log" || done_all=false
  done
  [[ "$done_all" == true ]] && { say "skip $TAG (all evals done)"; continue; }
  PDIR=$M2/ling-amendment2
  PRED=$(python3 -c "import json;print(json.load(open('$PDIR/$PLAN-plan.json'))['predicted_size_bytes'])")
  tmpout=$ART/ling-$TAG.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "quantize ling-$TAG (plan $PLAN) start predicted=$PRED lower=$LOW (tmpfs)"
    "$RT/llama-quantize" --imatrix "$IMX" --tensor-type-file "$PDIR/$PLAN-tensor-types.txt" \
      "$BF16" "$tmpout" "$LOW" > "$LOG/quantize-ling-$TAG.log" 2>&1
    [[ -f "$tmpout" ]] || { say "quantize ling-$TAG FAILED"; continue; }
  fi
  ACT=$(stat -c %s "$tmpout")
  # bailingmoe systematic offset (measured 6/6, see prereg amendment-2 note 2):
  # actual == predicted - 480 exactly; anything else is a hard fail.
  if [[ "$ACT" != "$((PRED - 480))" ]]; then
    say "ling-$TAG SIZE GATE FAIL actual=$ACT predicted=$PRED (expected $((PRED - 480))) — recorded as-is, not evaluated"
    record "ling-$TAG (SIZE FAIL actual=$ACT predicted=$PRED)" "$tmpout"
    rm -f "$tmpout"
    continue
  fi
  record "ling-$TAG" "$tmpout"
  eval_model "$tmpout" "ling-$TAG"
  rm -f "$tmpout"
  say "ling-$TAG ($PLAN) hash-recorded and released"
  n=$((n+1))
  if [[ $((n % 3)) -eq 0 ]]; then say "periodic record sync"; sync_state; fi
done
sync_state
say "================ AMENDMENT-2 DONE ================"
