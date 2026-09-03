#!/usr/bin/env bash
# M2 third dense-family calibration: google/gemma-4-E4B (dense text tower, DEV).
# Owner ruling 2026-09-02: Mistral-7B-v0.3 replaced by gemma-4-E4B (GPT approved
# with a text-tower hard gate — see results-gemma/text-tower-eligibility.json).
# v3 tmpfs-only hot loop, same discipline as run_ling_calibration.sh: the OS
# nvme is NTFS and has hard-panicked under sustained mixed read+write. NVMe is
# touched only for the one-time sequential copy in, small text-record syncs
# out, and TF-card record copies. Artifacts are hash-recorded then DELETED.
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
MODELDIR="/run/media/s117/OS/Models/gemma-4-E4B"
CORPUS="/run/media/s117/OS/Models/imatrix-calibration/APEX-imatrix-Small.txt"
QROOT=/dev/shm/m2-gemma
LOG=$QROOT/logs
REFS=$QROOT/refs
ART=$QROOT/artifacts
STATE=$QROOT/state
DISK_LOG=$M2/logs-gemma
DISK_STATE=$M2/results-gemma
DISK_REF=$M2/refs-gemma
TFDIR="/run/media/s117/TF 512G"
mkdir -p "$LOG" "$REFS" "$ART" "$STATE" "$DISK_LOG" "$DISK_STATE" "$DISK_REF"

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/gemma-progress.log"; }

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
  cp -f "$LOG"/gemma-progress.log "$LOG"/eval-*.log "$LOG"/quantize-*.log "$LOG"/ref-*.log "$LOG"/imatrix-*.log "$DISK_LOG/" 2>/dev/null
  cp -f "$STATE/artifact-manifest.txt" "$DISK_STATE/" 2>/dev/null
  # refs are keepers (they anchor the eval-v1 write path): sync to disk once
  for d in "${DOMAINS[@]}"; do
    [[ -s "$REFS/bf16-$d.kld" && ! -s "$DISK_REF/bf16-$d.kld" ]] && cp "$REFS/bf16-$d.kld" "$DISK_REF/" 2>/dev/null
  done
  sync
  if [[ -d "$TFDIR" && -w "$TFDIR" ]]; then
    mkdir -p "$TFDIR/fit-m2-gemma-records"
    cp -f "$LOG"/gemma-progress.log "$DISK_LOG"/eval-*.log "$STATE/artifact-manifest.txt" \
      "$TFDIR/fit-m2-gemma-records/" 2>/dev/null
    say "records also copied to TF card"
  fi
}

eval_model() {
  local gguf=$1 tag=$2 d log attempt
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then say "skip eval ${tag}/${d} (done)"; continue; fi
    say "eval ${tag}/${d} start"
    for attempt in 1 2 3; do
      rm -f "$log"
      "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
      grep -q "Mean.*KLD" "$log" && break
      say "eval ${tag}/${d} attempt $attempt produced no Mean KLD; backing off"
      sleep 20
    done
    if grep -q "Mean.*KLD" "$log"; then
      say "eval ${tag}/${d} OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')"
    else
      say "eval ${tag}/${d} FAILED after retries (no Mean KLD in log)"
    fi
    sleep 8
  done
}

ref_ok() { # file — exact b10666 _logits_ arithmetic: 20B header + tokens +
  # 255 rows per chunk of (n_vocab+4) uint16. Catches ENOSPC truncation,
  # which is silent (ofstream dies mid-row, process keeps printing).
  python3 - "$1" <<'PYEOF'
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

gen_ref() { # domain — verify row-completeness, not just non-emptiness
  local d=$1 log="$LOG/ref-gemma-$1.log" out="$REFS/bf16-$1.kld" attempt
  for attempt in 1 2 3 4; do
    say "generating ref $d (attempt $attempt)"
    rm -f "$out" "$log"
    "$RT/llama-perplexity" -m "$BF16" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence-base "$out" > "$log" 2>&1
    if ref_ok "$out" && grep -q "Final estimate" "$log"; then
      say "ref $d OK size=$(stat -c %s "$out")"
      return 0
    fi
    say "ref $d attempt $attempt invalid (size=$(stat -c %s "$out" 2>/dev/null || echo 0), ref_ok=false); backing off"
    sleep 20
  done
  say "ref $d FAILED after 4 attempts"
  return 1
}

say "================ gemma-4-E4B: tmpfs-only hot loop ================"

# ---- stage 0: text-tower hard gate (GPT ruling) must be green ----
ELIG="$DISK_STATE/text-tower-eligibility.json"
if [[ ! -s "$ELIG" ]] || ! python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r['eligible'] and r['vision_tensor_count']==0 and r['audio_tensor_count']==0 else 1)" "$ELIG"; then
  say "TEXT-TOWER ELIGIBILITY GATE NOT GREEN — pausing per GPT ruling"; exit 1
fi
say "text-tower eligibility gate: PASS"

# ---- stage 0b: resume ----
cp -f "$DISK_LOG"/eval-*.log "$LOG/" 2>/dev/null
cp -f "$DISK_LOG"/gemma-progress.log "$LOG/" 2>/dev/null
cp -f "$DISK_STATE/artifact-manifest.txt" "$STATE/" 2>/dev/null

# ---- stage 1: one-time sequential copy of heavy inputs into tmpfs ----
BF16=$QROOT/gemma-bf16.gguf
if [[ ! -f "$BF16" ]]; then
  say "copying BF16 into tmpfs (15G, sequential read)"
  cp "$MODELDIR/gemma-4-E4B-bf16.gguf" "$BF16" || { say "BF16 COPY FAILED"; exit 1; }
  sync
  say "BF16 in tmpfs sha=$(sha256sum "$BF16" | cut -c1-16)"
fi
CORPUS_LOCAL=$QROOT/apex-imatrix-small.txt
[[ -f "$CORPUS_LOCAL" ]] || { cp "$CORPUS" "$CORPUS_LOCAL" && sync; }
record "gemma-BF16" "$BF16"

# ---- stage 1a: imatrix (same APEX corpus lineage as granite/ling, 500x512) ----
IMX=$QROOT/gemma-imatrix.dat
if [[ ! -s "$IMX" ]]; then
  if [[ -s "$M2/results-gemma/gemma-imatrix.dat.sha" && -s "$DISK_STATE/gemma-imatrix.dat" ]]; then
    cp "$DISK_STATE/gemma-imatrix.dat" "$IMX"
  else
    say "generating imatrix (APEX corpus, 500 chunks x c512)"
    "$RT/llama-imatrix" -m "$BF16" -f "$CORPUS_LOCAL" -c 512 -ngl 99 \
      --chunks 500 -o "$IMX" > "$LOG/imatrix-gemma.log" 2>&1
    [[ -s "$IMX" ]] && { say "imatrix done size=$(stat -c %s "$IMX")"; cp "$IMX" "$DISK_STATE/" && sha256sum "$IMX" > "$DISK_STATE/gemma-imatrix.dat.sha"; } \
      || { say "imatrix FAILED"; sync_state; exit 1; }
  fi
fi
record "gemma-imatrix" "$IMX"

# ---- stage 1a2: low-bit override set (documented deviation) ----
# b10666's imatrix collection deterministically misses attn_k.weight for
# gemma4 blocks 24-41 (verified twice). IQ3_XXS/IQ2_XXS/IQ2_XS presets require
# imatrix entries for every tensor carrying those dst types, so those 18
# tensors are promoted to Q4_K via --tensor-type-file (the same fallback
# llama.cpp applies to imatrix-less tensors in low-bit presets). Recorded as
# an M2 amendment note; all other presets are pure.
OVR=$QROOT/lowbit-attnk-overrides.txt
python3 - "$IMX" "$OVR" <<'PYEOF'
import sys
sys.path.insert(0, "/run/media/s117/OS/FIT-GGUF/src")
from fit_gguf.imatrix import load_imatrix_profile
profile = load_imatrix_profile(sys.argv[1])
names = {e.name for e in profile.entries}
blocks = [b for b in range(64) if f"blk.{b}.attn_q.weight" in names
          and f"blk.{b}.attn_k.weight" not in names]
with open(sys.argv[2], "w") as f:
    for b in blocks:
        f.write(f"^blk\\.{b}\\.attn_k\\.weight$=Q4_K\n")
print(f"lowbit overrides: {len(blocks)} attn_k tensors -> Q4_K")
PYEOF
touch "$OVR"
# Empirical: any preset whose recipe assigns a requires_imatrix dst
# (IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/IQ1_*) to the 18 imatrix-less attn_k tensors
# must use the override file — judged per-tensor, not per-preset name.
LOWTYPES=" IQ2_XXS IQ2_XS IQ2_M IQ3_XXS IQ3_XS "

# ---- stage 1b: eval-v1 five-domain references from BF16 (write path) ----
# Space budget: refs ≈ 5 × (255 rows × (n_vocab+4) u16 × n_chunk) ≈ 21G for
# gemma4; BF16 15G may already be resident, so the requirement depends on it.
FREE_KB=$(df -k /dev/shm | tail -1 | awk '{print $4}')
NEED_KB=24576000
[[ -f "$BF16" ]] || NEED_KB=40960000
if [[ $FREE_KB -lt $NEED_KB ]]; then
  say "INSUFFICIENT tmpfs free space (${FREE_KB}K < ${NEED_KB}K) — aborting"; exit 1
fi
for d in "${DOMAINS[@]}"; do
  if [[ ! -s "$REFS/bf16-$d.kld" ]] || ! ref_ok "$REFS/bf16-$d.kld"; then
    src="$DISK_REF/bf16-$d.kld"
    if [[ -s "$src" ]] && ref_ok "$src"; then
      say "copying ref $d into tmpfs ($(stat -c %s "$src") bytes)"
      cp "$src" "$REFS/bf16-$d.kld" || { say "ref $d COPY FAILED"; exit 1; }
      sync
    else
      [[ -s "$src" ]] && say "ref $d on disk fails row-completeness — regenerating"
      gen_ref "$d" || { say "ref $d unrecoverable"; sync_state; exit 1; }
      sleep 8
    fi
  else
    say "ref $d already in tmpfs (verified)"
  fi
done
sync_state

# ---- stage 2: preset ladder (~12 points, GPT template) ----
say "== preset ladder =="
n=0
for QT in IQ2_XXS IQ2_XS IQ2_M IQ3_XXS IQ3_XS IQ3_M IQ4_XS Q3_K_M Q4_K_M Q5_K_M Q6_K Q8_0; do
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-gemma-$QT-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-gemma-$QT-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip $QT (all evals done)"; continue; fi
  tmpout=$ART/gemma-$QT.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "quantize gemma-$QT start (tmpfs)"
    if [[ "$LOWTYPES" == *" $QT "* ]]; then
      "$RT/llama-quantize" --imatrix "$IMX" --tensor-type-file "$OVR" "$BF16" "$tmpout" "$QT" > "$LOG/quantize-gemma-$QT.log" 2>&1
    else
      "$RT/llama-quantize" --imatrix "$IMX" "$BF16" "$tmpout" "$QT" > "$LOG/quantize-gemma-$QT.log" 2>&1
    fi
    [[ -f "$tmpout" ]] && say "quantize gemma-$QT done size=$(stat -c %s "$tmpout")" || { say "quantize gemma-$QT FAILED"; continue; }
  fi
  record "gemma-$QT" "$tmpout"
  eval_model "$tmpout" "gemma-$QT"
  rm -f "$tmpout"
  say "gemma-$QT artifact hash-recorded and released (tmpfs)"
  n=$((n+1))
  if [[ $((n % 4)) -eq 0 ]]; then say "periodic record sync"; sync_state; fi
done

sync_state
say "================ GEMMA PRESET LADDER DONE (FIT fill follows) ================"
