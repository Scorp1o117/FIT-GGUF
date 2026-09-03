#!/usr/bin/env bash
# M2 gemma FIT window-fill: 5 points to plug the preset-cliff-hollowed windows
# (PREREG-GEMMA §2; GPT template "preset rough curve, then FIT fills the
# hollowed windows"). Targets are linear crossings of the 12-point preset
# curve; planning runs through fit analyze/plan and quantize through the G2
# re-finalization gate. All hot I/O on tmpfs (driver still resident).
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
QROOT=/dev/shm/m2-gemma
LOG=$QROOT/logs
ART=$QROOT/artifacts
STATE=$QROOT/state
DISK_LOG=$M2/logs-gemma
DISK_STATE=$M2/results-gemma
TFDIR="/run/media/s117/TF 512G"
BF16=$QROOT/gemma-bf16.gguf
IMX=$QROOT/gemma-imatrix.dat
DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/gemma-progress.log"; }
sync_state() {
  sync
  cp -f "$LOG"/gemma-progress.log "$LOG"/eval-gemma-FIT-*.log "$LOG"/fit-*.log "$DISK_LOG/" 2>/dev/null
  cp -f "$STATE/artifact-manifest.txt" "$DISK_STATE/" 2>/dev/null
  if [[ -d "$TFDIR" && -w "$TFDIR" ]]; then
    mkdir -p "$TFDIR/fit-m2-gemma-records"
    cp -f "$LOG"/gemma-progress.log "$DISK_LOG"/eval-gemma-FIT-*.log "$STATE/artifact-manifest.txt" \
      "$TFDIR/fit-m2-gemma-records/" 2>/dev/null
  fi
}

record() {
  local name=$1 path=$2
  if grep -q "^$name  " "$STATE/artifact-manifest.txt" 2>/dev/null; then return 0; fi
  if [[ -f "$path" ]]; then
    printf '%s  %s  %s\n' "$name" "$(stat -c %s "$path")" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$STATE/artifact-manifest.txt"
  fi
}

eval_model() {
  local gguf=$1 tag=$2 d log attempt
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then continue; fi
    for attempt in 1 2 3; do
      rm -f "$log"
      "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$QROOT/refs/bf16-${d}.kld" > "$log" 2>&1
      grep -q "Mean.*KLD" "$log" && break
      sleep 15
    done
    grep -q "Mean.*KLD" "$log" && say "eval ${tag}/${d} OK" || say "eval ${tag}/${d} FAILED"
    sleep 5
  done
}

# ---- targets: linear crossings on the recorded curve ----
# Pass 2 anchors use the 17-point curve (pass-1 FIT points revealed strong
# non-linearity: kld@size lands LOWER than linear interpolation near Q5_K_M,
# so pass-1 G05/G10A/G10B fell below their windows; re-aim mid-window).
read -r G05B G10C G10D <<< "$(python3 - <<'PYEOF'
import json
d = json.load(open("experiments/2026-09-02-m2-topkl-calibration/results/candidate-calibration-v1.json"))
g = next(m for m in d["models"] if m["model"] == "gemma")
pts = sorted((p for p in g["points"] if p.get("macro_kld") is not None), key=lambda p: p["bytes"])
def crossing(t, lo, hi):
    a = next(p for p in pts if p["name"] == lo)
    b = next(p for p in pts if p["name"] == hi)
    f = (t - a["macro_kld"]) / (b["macro_kld"] - a["macro_kld"])
    return round(a["bytes"] + f * (b["bytes"] - a["bytes"]))
print(crossing(0.047, "FIT-G05", "Q5_K_M"),
      crossing(0.100, "Q4_K_M", "FIT-G10A"),
      crossing(0.110, "Q4_K_M", "FIT-G10A"))
PYEOF
)"
say "FIT pass-2 targets: G05B=$G05B G10C=$G10C G10D=$G10D"

declare -A PAIR=( [G05B]="Q5_K_M Q6_K" [G10C]="Q4_K_M Q5_K_M" [G10D]="Q4_K_M Q5_K_M" )
declare -A TARGET=( [G05B]="$G05B" [G10C]="$G10C" [G10D]="$G10D" )

for TAG in G05B G10C G10D; do
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-gemma-FIT-$TAG-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-gemma-FIT-$TAG-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip FIT-$TAG (done)"; continue; fi

  read -r LOWER UPPER <<< "${PAIR[$TAG]}"
  ANDIR="$M2/gemma-analysis-$LOWER-$UPPER"
  if [[ ! -s "$ANDIR/analysis.json" ]]; then
    say "fit analyze $LOWER->$UPPER"
    PYTHONPATH=src python3 -m fit_gguf.cli analyze --source "$BF16" --imatrix "$IMX" \
      --runtime "$RT" --out-dir "$ANDIR" --lower "$LOWER" --upper "$UPPER" \
      --imatrix-arg "$IMX" --skip-hash > "$LOG/fit-analyze-$LOWER-$UPPER.log" 2>&1 \
      || { say "fit analyze FAILED for $LOWER-$UPPER"; continue; }
  fi

  PREFIX="$M2/gemma-fit-$TAG"
  if [[ ! -s "${PREFIX}-plan.json" ]]; then
    say "fit plan FIT-$TAG target=${TARGET[$TAG]}"
    PYTHONPATH=src python3 -m fit_gguf.cli plan --analysis "$ANDIR/analysis.json" \
      --target-bytes "${TARGET[$TAG]}" --policy balanced --model-name "gemma-4-E4B-FIT-$TAG" \
      --out-prefix "$PREFIX" > "$LOG/fit-plan-$TAG.log" 2>&1 \
      || { say "fit plan FAILED for FIT-$TAG"; continue; }
  fi

  tmpout=$ART/gemma-FIT-$TAG.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "fit quantize FIT-$TAG (G2 gate)"
    PYTHONPATH=src python3 -m fit_gguf.cli quantize --analysis "$ANDIR/analysis.json" \
      --tensor-types "${PREFIX}-tensor-types.txt" --out "$tmpout" \
      --imatrix-arg "$IMX" > "$LOG/fit-quantize-$TAG.log" 2>&1 \
      || { say "fit quantize FAILED for FIT-$TAG (see fit-quantize-$TAG.log)"; continue; }
    say "FIT-$TAG G2 gate: $(grep 'G2:' "$LOG/fit-quantize-$TAG.log" | tail -1)"
  fi

  record "gemma-FIT-$TAG" "$tmpout"
  eval_model "$tmpout" "gemma-FIT-$TAG"
  rm -f "$tmpout"
  say "FIT-$TAG artifact hash-recorded and released (tmpfs)"
  sync_state
done

sync_state
say "================ GEMMA FIT FILL DONE ================"
