#!/usr/bin/env bash
# M2 third-family calibration: Ling-3.0-tiny-abliterated (bailingmoe, DEV).
# v3: ALL hot I/O on tmpfs (/dev/shm) — the OS nvme is NTFS and has hard-panicked
# three times under sustained mixed read+write. NVMe is touched only twice:
#   (1) one-time sequential copy of BF16/refs/state into tmpfs at start,
#   (2) small text-record syncs back every few points + at the end.
# GGUF artifacts are hash-recorded then DELETED after eval (deterministic
# pipeline can regenerate any of them; prereg §8 deletes them after acceptance).
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
M2=experiments/2026-09-02-m2-topkl-calibration
MODELDIR="/run/media/s117/OS/Models/Ling-3.0-tiny-abliterated-APEX-GGUF"
QROOT=/dev/shm/m2-ling
LOG=$QROOT/logs
REFS=$QROOT/refs
ART=$QROOT/artifacts
STATE=$QROOT/state
DISK_LOG=$M2/logs-ling
DISK_STATE=$M2/results-ling
mkdir -p "$LOG" "$REFS" "$ART" "$STATE" "$DISK_LOG" "$DISK_STATE"

DOMAINS=(wiki_test wiki_valid chinese code agent_chat)
declare -A SLICE=(
  [wiki_test]=eval-data/kl-eval-64k.txt
  [wiki_valid]=eval-data/kl-eval-valid-64k.txt
  [chinese]=eval-data/kl-eval-cn-64k.txt
  [code]=eval-data/kl-eval-code-64k.txt
  [agent_chat]=eval-data/kl-eval-agent-64k.txt
)

say() { echo "$(date +%m-%d\ %H:%M:%S)  $*" | tee -a "$LOG/ling-progress.log"; }

record() { # name path → manifest line in tmpfs state (deduped by name)
  local name=$1 path=$2
  if grep -q "^$name  " "$STATE/artifact-manifest.txt" 2>/dev/null; then return 0; fi
  if [[ -f "$path" ]]; then
    printf '%s  %s  %s\n' "$name" "$(stat -c %s "$path")" "$(sha256sum "$path" | cut -d' ' -f1)" >> "$STATE/artifact-manifest.txt"
  else
    printf '%s  MISSING\n' "$name" >> "$STATE/artifact-manifest.txt"
  fi
}

sync_state() { # small text records → disk (ntfs, one small write burst) + TF if mounted
  sync
  cp -f "$LOG"/ling-progress.log "$LOG"/eval-*.log "$LOG"/quantize-*.log "$LOG"/ref-*.log "$DISK_LOG/" 2>/dev/null
  cp -f "$STATE/artifact-manifest.txt" "$DISK_STATE/" 2>/dev/null
  sync
  if [[ -d /run/media/s117/TF && -w /run/media/s117/TF ]]; then
    mkdir -p /run/media/s117/TF/fit-m2-ling-records
    cp -f "$LOG"/ling-progress.log "$DISK_LOG/eval-*.log" "$STATE/artifact-manifest.txt" \
      /run/media/s117/TF/fit-m2-ling-records/ 2>/dev/null
    say "records also copied to TF card"
  fi
}

eval_model() { # gguf tag   (gguf lives in tmpfs)
  local gguf=$1 tag=$2 d log
  for d in "${DOMAINS[@]}"; do
    log="$LOG/eval-${tag}-${d}.log"
    if [[ -s "$log" ]] && grep -q "Mean.*KLD" "$log"; then say "skip eval ${tag}/${d} (done)"; continue; fi
    say "eval ${tag}/${d} start"
    "$RT/llama-perplexity" -m "$gguf" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
      --kl-divergence --kl-divergence-base "$REFS/bf16-${d}.kld" > "$log" 2>&1
    if grep -q "Mean.*KLD" "$log"; then
      say "eval ${tag}/${d} OK: $(grep 'Mean' "$log" | tail -1 | sed 's/^[[:space:]]*//')"
    else
      say "eval ${tag}/${d} FAILED (no Mean KLD in log)"
    fi
  done
}

say "================ v3: tmpfs-only hot loop ================"

# ---- stage 0: carry existing state/logs from disk into tmpfs (resume) ----
cp -f "$DISK_LOG"/eval-*.log "$LOG/" 2>/dev/null
cp -f "$DISK_STATE/artifact-manifest.txt" "$STATE/" 2>/dev/null
cp -f "$DISK_LOG"/ling-progress.log "$LOG/" 2>/dev/null

# ---- stage 1: one-time sequential copy of the heavy inputs into tmpfs ----
BF16=$QROOT/Ling-bf16.gguf
IMX=$QROOT/ling-imatrix.dat
if [[ ! -f "$BF16" ]]; then
  say "copying BF16 into tmpfs (15.8G, sequential read)"
  cp "$MODELDIR/Ling-3.0-tiny-abliterated-bf16.gguf" "$BF16" || { say "BF16 COPY FAILED"; exit 1; }
  sync
  say "BF16 in tmpfs sha=$(sha256sum "$BF16" | cut -c1-16)"
fi
if [[ ! -f "$IMX" ]]; then
  cp "$MODELDIR/Ling-3.0-tiny-abliterated-imatrix.dat" "$IMX" && sync
fi
for d in "${DOMAINS[@]}"; do
  if [[ ! -s "$REFS/bf16-$d.kld" ]]; then
    src="$M2/refs-ling/bf16-$d.kld"
    if [[ -s "$src" ]]; then
      say "copying ref $d into tmpfs ($(stat -c %s "$src") bytes)"
      cp "$src" "$REFS/bf16-$d.kld" || { say "ref $d COPY FAILED"; exit 1; }
      sync
    else
      say "ref $d MISSING on disk — will regenerate from tmpfs BF16"
      "$RT/llama-perplexity" -m "$BF16" -f "${SLICE[$d]}" -ngl 99 -t 16 -c 512 -b 512 \
        --kl-divergence-base "$REFS/bf16-${d}.kld" > "$LOG/ref-ling-$d.log" 2>&1
      [[ -s "$REFS/bf16-$d.kld" ]] && say "ref $d generated size=$(stat -c %s "$REFS/bf16-$d.kld")" || say "ref $d FAILED"
    fi
  else
    say "ref $d already in tmpfs"
  fi
done

# ---- stage 2: preset ladder (hot loop 100% tmpfs) ----
say "== preset ladder =="
n=0
for QT in IQ2_XXS IQ2_XS IQ2_S IQ2_M Q2_K_S Q2_K IQ3_XXS IQ3_XS Q3_K_S IQ3_S IQ3_M IQ4_XS Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K Q8_0; do
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-ling-$QT-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-ling-$QT-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip $QT (all evals done)"; continue; fi
  tmpout=$ART/ling-$QT.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "quantize ling-$QT as $QT start (tmpfs)"
    "$RT/llama-quantize" --imatrix "$IMX" "$BF16" "$tmpout" "$QT" > "$LOG/quantize-ling-$QT.log" 2>&1
    [[ -f "$tmpout" ]] && say "quantize ling-$QT done size=$(stat -c %s "$tmpout")" || { say "quantize ling-$QT FAILED"; continue; }
  fi
  record "ling-$QT" "$tmpout"
  eval_model "$tmpout" "ling-$QT"
  rm -f "$tmpout"
  say "ling-$QT artifact hash-recorded and released (tmpfs)"
  n=$((n+1))
  if [[ $((n % 4)) -eq 0 ]]; then say "periodic record sync"; sync_state; fi
done

# ---- stage 3: published APEX fidelity tiers (product-fit anchors) ----
say "== published APEX fidelity tiers =="
for APEX in Quality Balanced Compact Mini; do
  srcfile="$MODELDIR/Ling-3.0-tiny-abliterated-APEX-I-$APEX.gguf"
  [[ "$APEX" == "Mini" ]] && srcfile="$MODELDIR/Ling-3.0-tiny-abliterated-APEX-Mini.gguf"
  done_all=true
  for d in "${DOMAINS[@]}"; do
    [[ -s "$LOG/eval-ling-APEX-$APEX-$d.log" ]] && grep -q "Mean.*KLD" "$LOG/eval-ling-APEX-$APEX-$d.log" || done_all=false
  done
  if [[ "$done_all" == true ]]; then say "skip APEX-$APEX (all evals done)"; continue; fi
  if [[ ! -f "$srcfile" ]]; then say "APEX $APEX source MISSING"; continue; fi
  tmpout=$ART/ling-APEX-$APEX.gguf
  if [[ ! -f "$tmpout" ]]; then
    say "copy APEX-$APEX into tmpfs"
    cp "$srcfile" "$tmpout" || { say "APEX $APEX COPY FAILED"; continue; }
  fi
  record "ling-APEX-$APEX" "$tmpout"
  eval_model "$tmpout" "ling-APEX-$APEX"
  rm -f "$tmpout"
  say "ling-APEX-$APEX hash-recorded and released"
done

sync_state
say "================ LING CALIBRATION DONE ================"
