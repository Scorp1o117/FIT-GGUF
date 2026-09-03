#!/usr/bin/env bash
# M3 Table A "Quality @ Fixed Size": FIT v0.1 and v0.2 artifacts at the four
# native preset anchor sizes (GPT ruling: same byte budget for all three
# methods; anchors = the preset artifacts' actual sizes). v0.1 = balanced
# allocator; v0.2 = balanced + Refine C_role; both quantized through the G2
# re-finalization gate. Source read-only from NTFS; artifacts on tmpfs.
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
SRC=/run/media/s117/OS/Models/orcarouter-Qwen3.8-27B-Uncensored/Qwen3.8-27B-Uncensored-BF16.gguf
IMX=/run/media/s117/OS/FIT-GGUF/imatrix_unsloth.gguf
REFS=$M2/../2026-09-02-eval-v1/refs-orcarouter
PROFILE=profiles/refine-profile-qwen-hybrid-dev1.json
QROOT=/dev/shm/orca-a
LOG=$QROOT/logs
ART=$QROOT/artifacts
mkdir -p "$LOG" "$ART"

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/tableA-progress.log"; }

# anchors: tier -> preset name, actual byte size, bracketing preset pair
declare -A ANCHOR_SIZE=(
  [Q5KM]=19231100096
  [IQ4XS]=15082507456
  [IQ3S]=12419329216
  [IQ3XXS]=11186371776
)
declare -A PAIR=(
  [Q5KM]="Q4_K_M Q5_K_M"
  [IQ4XS]="IQ3_M IQ4_XS"
  [IQ3S]="Q3_K_S IQ3_S"
  [IQ3XXS]="Q2_K IQ3_XXS"
)

for ANCHOR in Q5KM IQ4XS IQ3S IQ3XXS; do
  for MODE in v01 v02; do
    TAG="FIT-A-$ANCHOR-$MODE"
    done_all=true
    for d in "${DOMAINS[@]}"; do
      [[ -s "$M2/logs/eval-orcarouter-$TAG-$d.log" ]] && grep -q "Mean.*KLD" "$M2/logs/eval-orcarouter-$TAG-$d.log" || done_all=false
    done
    if [[ "$done_all" == true ]]; then say "skip $TAG (done)"; continue; fi

    read -r LOWER UPPER <<< "${PAIR[$ANCHOR]}"
    ANDIR="$M2/orcarouter-analysis-$LOWER-$UPPER"
    if [[ ! -s "$ANDIR/analysis.json" ]]; then
      say "fit analyze $LOWER->$UPPER"
      PYTHONPATH=src python3 -m fit_gguf.cli analyze --source "$SRC" --imatrix "$IMX" \
        --runtime "$RT" --out-dir "$ANDIR" --lower "$LOWER" --upper "$UPPER" \
        --imatrix-arg "$IMX" --skip-hash > "$LOG/fit-analyze-$LOWER-$UPPER.log" 2>&1 \
        || { say "fit analyze FAILED $LOWER-$UPPER"; continue; }
    fi

    PREFIX="$M2/orcarouter-$TAG"
    if [[ ! -s "${PREFIX}-plan.json" ]]; then
      EXTRA=""
      [[ "$MODE" == "v02" ]] && EXTRA="--refine-profile $PROFILE"
      say "fit plan $TAG target=${ANCHOR_SIZE[$ANCHOR]} ($MODE)"
      PYTHONPATH=src python3 -m fit_gguf.cli plan --analysis "$ANDIR/analysis.json" \
        --target-bytes "${ANCHOR_SIZE[$ANCHOR]}" --policy balanced $EXTRA \
        --model-name "orcarouter-FIT-A-$ANCHOR-$MODE" \
        --out-prefix "$PREFIX" > "$LOG/fit-plan-$TAG.log" 2>&1 \
        || { say "fit plan FAILED $TAG"; continue; }
      say "$TAG planned: $(python3 -c "import json;r=json.load(open('${PREFIX}-plan.json'));print(r['predicted_size_bytes'], r['dominant_qtype'], 'refine' if r.get('refine_profile') else 'plain')")"
    fi

    tmpout=$ART/orcarouter-$TAG.gguf
    if [[ ! -f "$tmpout" ]]; then
      say "fit quantize $TAG (G2 gate)"
      PYTHONPATH=src python3 -m fit_gguf.cli quantize --analysis "$ANDIR/analysis.json" \
        --tensor-types "${PREFIX}-tensor-types.txt" --out "$tmpout" \
        --imatrix-arg "$IMX" > "$LOG/fit-quantize-$TAG.log" 2>&1 \
        || { say "fit quantize FAILED $TAG"; continue; }
      say "$TAG $(grep 'G2:' "$LOG/fit-quantize-$TAG.log" | tail -1)"
    fi

    eval_ok=true
    for d in "${DOMAINS[@]}"; do
      log="$M2/logs/eval-orcarouter-$TAG-$d.log"
      [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log" && continue
      for attempt in 1 2 3; do
        rm -f "$log"
        "$RT/llama-perplexity" -m "$tmpout" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
          --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
        grep -q "Mean.*KLD" "$log" && break
        say "eval $TAG/$d attempt $attempt failed; backing off"
        sleep 15
      done
      grep -q "Mean.*KLD" "$log" && say "eval $TAG/$d OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')" \
        || { say "eval $TAG/$d FAILED"; eval_ok=false; }
      sleep 5
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
done

say "================ TABLE A BUILD DONE ================"
