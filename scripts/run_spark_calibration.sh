#!/usr/bin/env bash
# Spark-X2.5-4B-abliterated (T615) FIT tier calibration: imatrix + five-domain
# eval-v1 references + preset ladder. Same tmpfs-only hot-loop discipline as
# run_gemma_calibration.sh (OS nvme is NTFS and has hard-panicked under
# sustained mixed read+write). The PR-branch runtime (llama.cpp #27868) is the
# only build with spark2_5 support; its perplexity.cpp is byte-identical to
# the frozen b10666 KL semantics (diffed 2026-09-06).
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=/home/s117/llama.cpp-spark-pr/build-rocm/bin
export LD_LIBRARY_PATH=$RT
EXP=experiments/2026-09-06-spark-x25-4tier
MODELDIR=/run/media/s117/OS/Models/Spark-X2.5-4B-abliterated-GGUF
BF16_DISK=$MODELDIR/Spark-X2.5-4B-abliterated-BF16.gguf
CORPUS=/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt
QROOT=/dev/shm/spark
LOG=$QROOT/logs
REFS=$QROOT/refs
ART=$QROOT/art
STATE=$QROOT/state
DISK_LOG=$EXP/logs
DISK_STATE=$EXP/state
DISK_REF=$EXP/refs
mkdir -p "$LOG" "$REFS" "$ART" "$STATE" "$DISK_LOG" "$DISK_STATE" "$DISK_REF" "$EXP/analysis"

# seed-compatible prefix: load_seeds() matches eval-<prefix><point>-<domain>.log
MP=spark-x25-4b-abliterated-

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)
LADDER=(IQ2_XXS IQ2_XS IQ2_M IQ3_XXS IQ3_XS IQ3_M IQ4_XS Q3_K_M Q4_K_M Q4_K_S Q5_K_S Q5_K_M Q6_K Q8_0 Q2_K)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/spark-progress.log"; }

record() {
  local name=$1 path=$2
  if grep -q "^$name  " "$STATE/artifact-manifest.txt" 2>/dev/null; then return 0; fi
  if [[ -f "$path" ]]; then
    printf '%s  %s  %s\n' "$name" "$(stat -c %s "$path")" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$STATE/artifact-manifest.txt"
  else
    printf '%s  MISSING\n' "$name" >> "$STATE/artifact-manifest.txt"
  fi
}

sync_state() {
  sync
  cp -f "$LOG"/spark-progress.log "$LOG"/eval-*.log "$LOG"/quantize-*.log "$LOG"/ref-*.log "$LOG"/imatrix-*.log "$DISK_LOG/" 2>/dev/null
  cp -f "$STATE/artifact-manifest.txt" "$DISK_STATE/" 2>/dev/null
  for d in "${DOMAINS[@]}"; do
    [[ -s "$REFS/bf16-$d.kld" && ! -s "$DISK_REF/bf16-$d.kld" ]] && cp "$REFS/bf16-$d.kld" "$DISK_REF/" 2>/dev/null
  done
  sync
}

eval_model() { # gguf tag
  local gguf=$1 tag=$2 d log attempt
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then continue; fi
    for attempt in 1 2 3; do
      rm -f "$log"
      "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
      grep -q "Mean.*KLD" "$log" && break
      say "eval ${tag}/${d} attempt $attempt no Mean KLD; backoff"
      sleep 15
    done
    grep -q "Mean.*KLD" "$log" || say "eval ${tag}/${d} FAILED after retries"
    sleep 5
  done
}

ref_ok() { python3 - "$1" <<'PYEOF'
import os, struct, sys
p = sys.argv[1]
try:
    size = os.path.getsize(p)
    with open(p, "rb") as f:
        head = f.read(20)
    if len(head) < 20 or head[:8] != b"_logits_":
        sys.exit(1)
    n_ctx, n_vocab, n_chunk = struct.unpack("<III", head[8:20])
    nv = 2 * ((n_vocab + 1) // 2) + 4
    expect = 20 + n_chunk * n_ctx * 4 + n_chunk * (n_ctx - 1 - n_ctx // 2) * nv * 2
    sys.exit(0 if size == expect else 1)
except Exception:
    sys.exit(1)
PYEOF
}

gen_ref() { # domain
  local d=$1 log="$LOG/ref-spark-$1.log" out="$REFS/bf16-$1.kld" attempt
  for attempt in 1 2 3 4; do
    say "generating ref $d (attempt $attempt)"
    rm -f "$out" "$log"
    "$RT/llama-perplexity" -m "$BF16" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence-base "$out" > "$log" 2>&1
    if ref_ok "$out" && grep -q "Final estimate" "$log"; then
      say "ref $d OK size=$(stat -c %s "$out")"
      return 0
    fi
    say "ref $d attempt $attempt invalid; backoff"
    sleep 15
  done
  say "ref $d FAILED after 4 attempts"
  return 1
}

say "================ spark-x25-4b (T615): tmpfs-only hot loop ================"

# resume
cp -f "$DISK_LOG"/eval-*.log "$LOG/" 2>/dev/null
cp -f "$DISK_LOG"/spark-progress.log "$LOG/" 2>/dev/null
cp -f "$DISK_STATE/artifact-manifest.txt" "$STATE/" 2>/dev/null
for d in "${DOMAINS[@]}"; do
  [[ -s "$DISK_REF/bf16-$d.kld" && ! -s "$REFS/bf16-$d.kld" ]] && cp "$DISK_REF/bf16-$d.kld" "$REFS/" 2>/dev/null
done

# ---- stage 1: BF16 into tmpfs (one sequential read) ----
BF16=$QROOT/spark-bf16.gguf
if [[ ! -f "$BF16" ]]; then
  [[ -s "$BF16_DISK" ]] || { say "BF16 GGUF MISSING on disk — convert first"; exit 1; }
  say "copying BF16 into tmpfs (8.2G, sequential)"
  cp "$BF16_DISK" "$BF16" || { say "BF16 COPY FAILED"; exit 1; }
  sync
fi
record "spark-BF16" "$BF16"
say "BF16 sha=$(sha256sum "$BF16" | cut -d' ' -f1)"

# ---- stage 1a: imatrix (APEX corpus, 500 chunks x c512) ----
IMX=$QROOT/imatrix.gguf
IMX_DISK=$EXP/spark-imatrix.gguf
if [[ ! -s "$IMX" ]]; then
  if [[ -s "$IMX_DISK" ]]; then
    cp "$IMX_DISK" "$IMX"
  else
    say "generating imatrix (APEX corpus, 500 chunks x c512)"
    "$RT/llama-imatrix" -m "$BF16" -f "$CORPUS" -c 512 -ngl 99 \
      --chunks 500 -o "$IMX" > "$LOG/imatrix-spark.log" 2>&1
    [[ -s "$IMX" ]] && { cp "$IMX" "$IMX_DISK"; sha256sum "$IMX" > "$DISK_STATE/spark-imatrix.gguf.sha"; } \
      || { say "imatrix FAILED"; sync_state; exit 1; }
  fi
fi
record "spark-imatrix" "$IMX"
say "imatrix coverage check"
/home/s117/unsloth_env/bin/python - "$IMX" "$BF16" 2>&1 | tee -a "$LOG/spark-progress.log" <<'PYEOF'
import sys
sys.path.insert(0, "/run/media/s117/OS/FIT-GGUF/src")
from pathlib import Path
from fit_gguf.imatrix import load_imatrix_profile
from fit_gguf.gguf import read_gguf_layout
imx = load_imatrix_profile(Path(sys.argv[1]))
have = {e.name for e in imx.entries}
layout = read_gguf_layout(Path(sys.argv[2]))
missing = [t.name for t in layout.tensors if t.name not in have and "embd" not in t.name]
print(f"imatrix entries={len(have)}  tensors_missing_imatrix={len(missing)}")
for name in missing[:20]:
    print("  MISSING:", name)
PYEOF

# ---- stage 1b: five-domain references from BF16 (eval-v1 write path) ----
FREE_KB=$(df -k /dev/shm | tail -1 | awk '{print $4}')
[[ $FREE_KB -lt 30000000 ]] && { say "INSUFFICIENT tmpfs (${FREE_KB}K) — abort"; exit 1; }
for d in "${DOMAINS[@]}"; do
  if [[ ! -s "$REFS/bf16-$d.kld" ]] || ! ref_ok "$REFS/bf16-$d.kld"; then
    gen_ref "$d" || { say "ref $d unrecoverable"; sync_state; exit 1; }
    sleep 5
  else
    say "ref $d present (verified)"
  fi
done
sync_state

# ---- stage 2: preset ladder (raw quantize + eval-v1 five-domain) ----
say "== preset ladder =="
n=0
for QT in "${LADDER[@]}"; do
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-${MP}${QT}-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-${MP}${QT}-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip $QT (done)"; continue; fi
  tmpout=$ART/spark-$QT.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "quantize $QT start (tmpfs)"
    rm -f "$tmpout"
    "$RT/llama-quantize" --imatrix "$IMX" "$BF16" "$tmpout" "$QT" > "$LOG/quantize-spark-$QT.log" 2>&1
    [[ -f "$tmpout" ]] && say "quantize $QT done size=$(stat -c %s "$tmpout")" \
      || { say "quantize $QT FAILED"; continue; }
  fi
  record "${MP}${QT}" "$tmpout"
  eval_model "$tmpout" "${MP}${QT}"
  rm -f "$tmpout"
  n=$((n+1))
  if [[ $((n % 4)) -eq 0 ]]; then say "periodic sync"; sync_state; fi
done

sync_state
say "================ SPARK PRESET LADDER DONE (floors + windows next) ================"
