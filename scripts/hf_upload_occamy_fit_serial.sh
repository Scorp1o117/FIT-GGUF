#!/usr/bin/env bash
# Serial HF upload for occamy-1.0-abliterated-FIT-GGUF.
#
# Same shape as hf_upload_fit_serial.sh, which survived the Qwen3.8-27B release:
# DIRECT connection (the proxy was the source of every drop on that run), one
# file at a time, xet upload concurrency pinned to 1 so the chunks are truly
# serial, and 20 attempts per item before giving up on it. Small provenance
# first, then the GGUFs in ascending size so the repository is coherent long
# before the 64 GiB BF16 lands.
set -u
SRC=/run/media/s117/OS/Models/occamy-1.0-abliterated-FIT-GGUF
REPO=SC117/occamy-1.0-abliterated-FIT-GGUF
LOG=/run/media/s117/OS/FIT-GGUF/hf-upload-occamy-fit.log

cd "$SRC"
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY all_proxy
# The bracketed IPv6 form `[::1]` in NO_PROXY makes httpx raise
# "Invalid port: ':1]'" before a single request leaves the machine — it is not a
# proxy problem and unsetting the proxies does not fix it. Bare `::1` is fine.
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export HF_XET_CLIENT_READ_TIMEOUT=30
export HF_XET_FIXED_UPLOAD_CONCURRENCY=1

items=(README.md README.zh-CN.md SHA256SUMS.txt results)
while IFS= read -r f; do items+=("$f"); done < <(ls -SrS *.gguf)

echo "=== SERIAL UPLOAD START $(date) ===" >> "$LOG"
echo "=== items: ${#items[@]} ===" >> "$LOG"

fail_total=0
for item in "${items[@]}"; do
  ok=0
  for attempt in $(seq 1 20); do
    echo "=== [$item] attempt $attempt $(date '+%m-%d %H:%M:%S') ===" >> "$LOG"
    /home/s117/.local/bin/hf upload "$REPO" "$item" "$item" --repo-type model >> "$LOG" 2>&1 \
      && { ok=1; break; }
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
