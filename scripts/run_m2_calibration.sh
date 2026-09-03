#!/usr/bin/env bash
# M2 Top|KL candidate calibration — serial stage runner.
# Preregistration: experiments/2026-09-02-m2-topkl-calibration/PREREGISTRATION.md
# Failures are recorded as-is and do NOT abort the run (no set -e).
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
LOG=$M2/logs
OUT=artifacts/fit/m2-calib
M16=experiments/2026-08-29-m16-granite-reveal
mkdir -p "$LOG" "$OUT" "$M2/results" "$M2/fit14g"

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/driver-progress.log"; }

record() { # name path  → sha+size manifest line (deduped by name)
  local name=$1 path=$2
  if grep -q "^$name  " "$M2/results/artifact-manifest.txt" 2>/dev/null; then return 0; fi
  if [[ -f "$path" ]]; then
    printf '%s  %s  %s\n' "$name" "$(stat -c %s "$path")" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$M2/results/artifact-manifest.txt"
  else
    printf '%s  MISSING\n' "$name" >> "$M2/results/artifact-manifest.txt"
  fi
}

eval_model() { # gguf refdir tag
  local gguf=$1 refdir=$2 tag=$3 d log
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then say "skip eval ${tag}/${d} (done)"; continue; fi
    say "eval ${tag}/${d} start"
    "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence --kl-divergence-base "$refdir/bf16-${d}.kld" > "$log" 2>&1
    if grep -q "Mean.*KLD" "$log"; then
      say "eval ${tag}/${d} OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')"
    else
      say "eval ${tag}/${d} FAILED (no Mean KLD in log)"
    fi
  done
}

quantize_one() { # src imx out qtype [typefile]
  local src=$1 imx=$2 out=$3 qtype=$4 tf=${5:-}
  if [[ -f "$out" ]]; then say "quantize skip $(basename "$out") (exists)"; return 0; fi
  say "quantize $(basename "$out") as $qtype start"
  if [[ -n "$tf" ]]; then
    "$RT/llama-quantize" --imatrix "$imx" --tensor-type-file "$tf" "$src" "$out" "$qtype" > "$LOG/quantize-$(basename "$out" .gguf).log" 2>&1
  else
    "$RT/llama-quantize" --imatrix "$imx" "$src" "$out" "$qtype" > "$LOG/quantize-$(basename "$out" .gguf).log" 2>&1
  fi
  if [[ -f "$out" ]]; then say "quantize $(basename "$out") done size=$(stat -c %s "$out")"; else say "quantize $(basename "$out") FAILED"; fi
}

########################################
say "================ STAGE 1: orcarouter ================"
ORB_SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
ORB_IMX=imatrix_unsloth.gguf
ORB_REF=experiments/2026-09-02-eval-v1/refs-orcarouter

for QT in Q4_K_S Q4_K_M Q5_K_S Q5_K_M; do
  out=$OUT/orcarouter-$QT.gguf
  quantize_one "$ORB_SRC" "$ORB_IMX" "$out" "$QT"
  record "orcarouter-$QT" "$out"
  [[ -f "$out" ]] && eval_model "$out" "$ORB_REF" "orcarouter-$QT"
done

# ---- FIT-14G (calibration-only, pair Q3_K_L->IQ4_XS, policy balanced) ----
say "plan FIT-14G (target 15032385536)"
PYTHONPATH=src python3 -m fit_gguf.cli plan \
  --analysis experiments/2026-08-29-p4-release-batch/tiers/_analysis/Q3_K_L-IQ4_XS/analysis.json \
  --target-bytes 15032385536 --policy balanced \
  --model-name orcarouter-Qwen3.8-27B-Uncensored \
  --out-prefix "$M2/fit14g/fit14g" > "$LOG/plan-orcarouter-FIT-14G.log" 2>&1
PLAN=$M2/fit14g/fit14g-plan.json
if [[ -f "$PLAN" ]]; then
  PRED=$(python3 -c "import json;print(json.load(open('$PLAN'))['predicted_size_bytes'])")
  LOW=$(python3 -c "import json;print(json.load(open('$PLAN'))['lower_preset'])")
  say "FIT-14G planned predicted=$PRED lower=$LOW"
  out=$OUT/orcarouter-FIT-14G.gguf
  quantize_one "$ORB_SRC" "$ORB_IMX" "$out" "$LOW" "$M2/fit14g/fit14g-tensor-types.txt"
  if [[ -f "$out" ]]; then
    ACT=$(stat -c %s "$out")
    if [[ "$ACT" == "$PRED" ]]; then
      record "orcarouter-FIT-14G" "$out"
      eval_model "$out" "$ORB_REF" "orcarouter-FIT-14G"
    else
      say "FIT-14G SIZE GATE FAIL actual=$ACT predicted=$PRED — recorded as-is, not evaluated"
      record "orcarouter-FIT-14G (SIZE FAIL actual=$ACT predicted=$PRED)" "$out"
    fi
  fi
else
  say "FIT-14G PLAN FAILED — recorded as-is"
fi

########################################
say "================ STAGE 2: granite ================"
GRA_SRC=artifacts/source/granite-4.2-8b-BF16.gguf
GRA_IMX=imatrix-granite-apex-c512.gguf
GRA_REF=$M2/refs-granite
mkdir -p "$GRA_REF"

if [[ ! -f "$GRA_SRC" ]]; then
  if [[ ! -f /tmp/llamacpp-master/convert_hf_to_gguf.py ]]; then
    say "fetching upstream converter (sparse)"
    rm -rf /tmp/llamacpp-master
    https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 \
      git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp /tmp/llamacpp-master \
      > "$LOG/clone-converter.log" 2>&1
    (cd /tmp/llamacpp-master && git sparse-checkout set --skip-checks gguf-py convert_hf_to_gguf.py) >> "$LOG/clone-converter.log" 2>&1
  fi
  git -C /tmp/llamacpp-master rev-parse HEAD > "$LOG/granite-converter-commit.txt" 2>/dev/null
  say "converting granite BF16"
  PYTHONPATH=/tmp/llamacpp-master/gguf-py /home/s117/unsloth_env/bin/python3 \
    /tmp/llamacpp-master/convert_hf_to_gguf.py /run/media/s117/OS/Models/granite-4.2-8b \
    --outfile "$GRA_SRC" --outtype bf16 > "$LOG/convert-granite-bf16.log" 2>&1
fi
if [[ -f "$GRA_SRC" ]]; then
  GSHA=$(sha256sum "$GRA_SRC" | cut -d' ' -f1)
  echo "$GSHA  granite-4.2-8b-BF16.gguf (M2 re-conversion)" > "$LOG/granite-bf16-sha.txt"
  say "granite BF16 sha=$GSHA"
  if [[ "$GSHA" == d82690e0dc827f2c43effeb3d489f572afbe2541fa8b4895b0d958ce473925a6 ]]; then
    say "granite BF16 == m16 recorded sha → bit-reproduction"
  else
    say "granite BF16 differs from m16 d82690e0… → honest deviation recorded, pipeline proceeds"
  fi
else
  say "granite BF16 CONVERSION FAILED — granite stage cannot proceed"
  exit 1
fi

say "granite references (two-phase: no --kl-divergence)"
for d in "${DOMAINS[@]}"; do
  if [[ -s "$GRA_REF/bf16-$d.kld" ]]; then say "skip ref $d (exists)"; continue; fi
  say "ref $d start"
  "$RT/llama-perplexity" -m "$GRA_SRC" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
    --kl-divergence-base "$GRA_REF/bf16-${d}.kld" > "$LOG/ref-granite-$d.log" 2>&1
  [[ -s "$GRA_REF/bf16-$d.kld" ]] && say "ref $d OK size=$(stat -c %s "$GRA_REF/bf16-$d.kld")" || say "ref $d FAILED"
done

for QT in IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M Q2_K_S Q2_K IQ3_XXS IQ3_XS Q3_K_S IQ3_S IQ3_M IQ4_XS Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0; do
  out=$OUT/granite-$QT.gguf
  quantize_one "$GRA_SRC" "$GRA_IMX" "$out" "$QT"
  record "granite-$QT" "$out"
  [[ -f "$out" ]] && eval_model "$out" "$GRA_REF" "granite-$QT"
done

# FIT points, recipes + expected sizes pinned from m16 (zero-tolerance gates)
quantize_fit() { # name expect tf  (split locals: single-line local+bound ref fails under set -u)
  local name=$1 expect=$2 tf=$3
  local out=$OUT/granite-$name.gguf
  quantize_one "$GRA_SRC" "$GRA_IMX" "$out" "IQ3_M" "$M16/$tf"
  if [[ -f "$out" ]]; then
    local act
    act=$(stat -c %s "$out")
    if [[ "$act" == "$expect" ]]; then
      record "granite-$name" "$out"
      eval_model "$out" "$GRA_REF" "granite-$name"
    else
      say "granite-$name SIZE GATE FAIL actual=$act expected=$expect — recorded as-is, not evaluated"
      record "granite-$name (SIZE FAIL actual=$act expected=$expect)" "$out"
    fi
  fi
}
quantize_fit O-FIT25 4271391104 o-fit25-tensor-types.txt
quantize_fit B-FIT25 4271931776 b-fit25-tensor-types.txt
quantize_fit O-FIT50 4454351232 o-fit50-tensor-types.txt
quantize_fit B-FIT50 4454564224 b-fit50-tensor-types.txt
quantize_fit O-FIT75 4637311360 o-fit75-tensor-types.txt
quantize_fit B-FIT75 4637311360 b-fit75-tensor-types.txt

say "================ M2 DRIVER DONE ================"

########################################
# STAGE 3: amendment-1 (see PREREGISTRATION §9) — Quality window coverage repair
say "================ STAGE 3: amendment-1 ================"
ORB_SRC=artifacts/source/orcarouter-Qwen3.8-27B-Uncensored-BF16.gguf
ORB_IMX=imatrix_unsloth.gguf
ORB_REF=experiments/2026-09-02-eval-v1/refs-orcarouter
ANA=$M2/analysis-Q4_K_M-Q5_K_S/analysis.json

for spec in "M2A 17800000000" "M2B 18400000000"; do
  set -- $spec
  NAME=$1; TGT=$2
  say "plan amendment $NAME (target $TGT)"
  mkdir -p "$M2/amendment-$NAME"
  PYTHONPATH=src python3 -m fit_gguf.cli plan \
    --analysis "$ANA" --target-bytes "$TGT" --policy balanced \
    --model-name orcarouter-Qwen3.8-27B-Uncensored \
    --out-prefix "$M2/amendment-$NAME/$NAME" > "$LOG/plan-orcarouter-$NAME.log" 2>&1
  PLAN=$M2/amendment-$NAME/$NAME-plan.json
  if [[ ! -f "$PLAN" ]]; then say "amendment $NAME PLAN FAILED — recorded as-is"; continue; fi
  PRED=$(python3 -c "import json;print(json.load(open('$PLAN'))['predicted_size_bytes'])")
  LOW=$(python3 -c "import json;print(json.load(open('$PLAN'))['lower_preset'])")
  say "amendment $NAME planned predicted=$PRED lower=$LOW"
  out=$OUT/orcarouter-$NAME.gguf
  quantize_one "$ORB_SRC" "$ORB_IMX" "$out" "$LOW" "$M2/amendment-$NAME/$NAME-tensor-types.txt"
  [[ -f "$out" ]] || continue
  ACT=$(stat -c %s "$out")
  if [[ "$ACT" == "$PRED" ]]; then
    record "orcarouter-$NAME" "$out"
    eval_model "$out" "$ORB_REF" "orcarouter-$NAME"
  else
    say "amendment $NAME SIZE GATE FAIL actual=$ACT predicted=$PRED — recorded as-is, not evaluated"
    record "orcarouter-$NAME (SIZE FAIL actual=$ACT predicted=$PRED)" "$out"
  fi
done
say "================ M2 ALL STAGES DONE ==============="
