#!/usr/bin/env bash
# HF upload with auto-retry: xet chunk dedup makes each retry incremental.
cd /run/media/s117/OS/FIT-GGUF/Qwen3.8-27B-Uncensored-FIT-GGUF
export HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897
export HF_XET_CLIENT_READ_TIMEOUT=30
LOG=/run/media/s117/OS/FIT-GGUF/hf-upload-fit.log
for i in $(seq 1 40); do
  echo "=== attempt $i $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
  /home/s117/.local/bin/hf upload SC117/Qwen3.8-27B-Uncensored-FIT-GGUF . . --repo-type model >> "$LOG" 2>&1 && {
    echo "=== UPLOAD COMPLETE $(date) ===" >> "$LOG"; exit 0; }
  echo "=== attempt $i failed, retrying in 60s ===" >> "$LOG"
  sleep 60
done
echo "=== GAVE UP after 40 attempts $(date) ===" >> "$LOG"
exit 1
