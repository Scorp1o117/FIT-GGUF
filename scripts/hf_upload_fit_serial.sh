#!/usr/bin/env bash
# Serial HF upload v3: DIRECT connection (proxy tested unnecessary for HF and
# was the source of every drop), one file at a time, xet upload concurrency
# fixed to 1 (truly serial chunks). Small provenance first, GGUFs ascending.
set -u
cd /run/media/s117/OS/FIT-GGUF/Qwen3.8-27B-Uncensored-FIT-GGUF
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY all_proxy
export HF_XET_CLIENT_READ_TIMEOUT=30
export HF_XET_FIXED_UPLOAD_CONCURRENCY=1
LOG=/run/media/s117/OS/FIT-GGUF/hf-upload-fit.log
REPO=SC117/Qwen3.8-27B-Uncensored-FIT-GGUF

items=(README.md README.zh-CN.md assets fit-plans results)
while IFS= read -r f; do items+=("$f"); done < <(ls -SrS *.gguf)

fail_total=0
for item in "${items[@]}"; do
  ok=0
  for attempt in $(seq 1 20); do
    echo "=== [$item] attempt $attempt $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
    /home/s117/.local/bin/hf upload "$REPO" "$item" "$item" --repo-type model >> "$LOG" 2>&1 && { ok=1; break; }
    echo "=== [$item] attempt $attempt failed, retry in 60s ===" >> "$LOG"
    sleep 60
  done
  if [[ $ok == 1 ]]; then
    echo "=== [$item] DONE $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
  else
    echo "=== [$item] GAVE UP after 20 attempts ===" >> "$LOG"
    fail_total=$((fail_total + 1))
  fi
done

if (( fail_total == 0 )); then
  echo "=== SERIAL UPLOAD COMPLETE $(date) ===" >> "$LOG"
  exit 0
fi
echo "=== SERIAL UPLOAD FINISHED WITH $fail_total FAILED ITEM(S) $(date) ===" >> "$LOG"
exit 1
