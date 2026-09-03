#!/usr/bin/env bash
# M6b band-conditional Refine validation on orcarouter-Qwen3.8-27B-Uncensored.
#
# Gate A (Fixed Size, no regression vs bootstrap-v0 at the same bytes):
#   Quality / Balanced crossing targets with the band profile — the Compact and
#   Mini crossing recipes came out byte-identical to bootstrap (v2b plan diff=0),
#   so their Gate A is an identity pass (same artifact, same KL).
# Gate B (Fixed Fidelity, Size_v2b <= Size_v0.1 at the tier KL anchor):
#   Balanced @ v0.1 FIT-13G bytes (13,956,046,016), window IQ3_M->IQ4_XS (same
#   window as v0.1); Mini is an identity pass (bootstrap 10.35G kld 0.1924
#   already <= v0.1 10.42G); Compact @ v0.1 FIT-11.5G bytes (12,277,279,936) in
#   the IQ3_XS->IQ3_S window — the same window as v0.1's artifact, because the
#   v0.2 Compact regression turned out to be confounded by its Q3_K_S floor
#   (poison preset). Two controls isolate the confound:
#     - v1 stack (no refine) in the same window/target
#     - band profile in the Q3_K_S->IQ3_S window at the same target
#
# Hot I/O on tmpfs; artifacts hash-recorded then deleted after eval. Spelling
# discipline: the model prefix is "orcarouter" everywhere (M2 lost four eval
# sets to an "orcorouter" typo).
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
SRC=/run/media/s117/OS/Models/orcarouter-Qwen3.8-27B-Uncensored/Qwen3.8-27B-Uncensored-BF16.gguf
IMX=/run/media/s117/OS/FIT-GGUF/imatrix_unsloth.gguf
REFS=$M2/../2026-09-02-eval-v1/refs-orcarouter
QROOT=/dev/shm/orca-m6b
LOG=$QROOT/logs
ART=$QROOT/artifacts
mkdir -p "$LOG" "$ART"
BAND=profiles/refine-profile-qwen-hybrid-band-v1.json

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/m6b-progress.log"; }

# name | lower | upper | target | profile(none = v1 control) | tag
RUNS=(
  "CompactFF  IQ3_XS IQ3_S 12277279936 $BAND FIT-V2B-CompactFF"
  "BalancedFF IQ3_M  IQ4_XS 13956046016 $BAND FIT-V2B-BalancedFF"
  "Quality    Q4_K_M Q5_K_M 17857895277 $BAND FIT-V2B-Quality"
  "Balanced   IQ3_M  IQ4_XS 13901839675 $BAND FIT-V2B-Balanced"
  "CompactFFC IQ3_XS IQ3_S 12277279936 none FIT-V1B-CompactFF"
  "CompactFFQ Q3_K_S IQ3_S 12277279936 $BAND FIT-V2B-CompactFFQ3"
)

eval_model() { # gguf tag
  local gguf=$1 tag=$2 d log
  for d in "${DOMAINS[@]}"; do
    log="$M2/logs/eval-orcarouter-$tag-$d.log"
    [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" && continue
    for attempt in 1 2 3; do
      rm -f "$log"
      "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
      grep -q "Mean.*KLD" "$log" && break
      say "eval $tag/$d attempt $attempt failed; backing off"
      sleep 15
    done
    grep -q "Mean.*KLD" "$log" \
      && say "eval $tag/$d OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')" \
      || say "eval $tag/$d FAILED"
    sleep 5
  done
}

for RUN in "${RUNS[@]}"; do
  read -r NAME LOWER UPPER TARGET PROFILE TAG <<< "$RUN"
  done_all=true
  for d in "${DOMAINS[@]}"; do
    log="$M2/logs/eval-orcarouter-$TAG-$d.log"
    [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip $TAG (done)"; continue; fi

  ANDIR="$M2/orcarouter-analysis-$LOWER-$UPPER"
  if [[ ! -s "$ANDIR/analysis.json" ]]; then
    say "fit analyze $LOWER->$UPPER"
    PYTHONPATH=src python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
      --runtime "$RT" --out-dir "$ANDIR" --lower "$LOWER" --upper "$UPPER" \
      --imatrix-arg "$IMX" --skip-hash > "$LOG/fit-analyze-$LOWER-$UPPER.log" 2>&1 \
      || { say "fit analyze FAILED $LOWER-$UPPER"; continue; }
  fi

  PREFIX="$M2/orcarouter-fit-$NAME"
  if [[ ! -s "${PREFIX}-plan.json" ]]; then
    PROFILE_ARGS=()
    [[ "$PROFILE" != "none" ]] && PROFILE_ARGS=(--refine-profile "$PROFILE")
    say "fit plan $NAME target=$TARGET profile=${PROFILE}"
    PYTHONPATH=src python3 -m fit_gguf.cli plan --analysis "$ANDIR/analysis.json" \
      --target-bytes "$TARGET" --policy balanced "${PROFILE_ARGS[@]}" \
      --model-name "orcarouter-Qwen3.8-27B-Uncensored-$TAG" \
      --out-prefix "$PREFIX" > "$LOG/fit-plan-$NAME.log" 2>&1 \
      || { say "fit plan FAILED $NAME"; continue; }
    say "$NAME planned: $(python3 -c "import json;r=json.load(open('${PREFIX}-plan.json'));print(r['predicted_size_bytes'], r['dominant_qtype'])")"
  fi

  tmpout=$ART/orcarouter-$TAG.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "fit quantize $NAME (G2 gate)"
    PYTHONPATH=src python3 -m fit_gguf.cli quantize --analysis "$ANDIR/analysis.json" \
      --tensor-types "${PREFIX}-tensor-types.txt" --out "$tmpout" \
      --imatrix-arg "$IMX" > "$LOG/fit-quantize-$NAME.log" 2>&1 \
      || { say "fit quantize FAILED $NAME"; continue; }
    say "$NAME $(grep 'G2:' "$LOG/fit-quantize-$NAME.log" | tail -1)"
  fi

  eval_model "$tmpout" "$TAG"

  eval_ok=true
  for d in "${DOMAINS[@]}"; do
    log="$M2/logs/eval-orcarouter-$TAG-$d.log"
    [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" || eval_ok=false
  done
  if [[ "$eval_ok" == true ]]; then
    if ! grep -q "^orcarouter-$TAG  " "$M2/results/artifact-manifest.txt" 2>/dev/null; then
      printf 'orcarouter-%s  %s  %s\n' "$TAG" "$(stat -c %s "$tmpout")" "$(sha256sum "$tmpout" | cut -d' ' -f1)" >> "$M2/results/artifact-manifest.txt"
    fi
    rm -f "$tmpout"
    say "$TAG hash-recorded and released (tmpfs)"
  else
    say "$TAG evals incomplete — artifact kept for resume"
  fi
  sync
done

say "================ M6B GATE RUNS DONE ================"
