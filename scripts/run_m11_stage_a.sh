#!/usr/bin/env bash
# M11 stage A: verify BF16 provenance and generate five holdout reference logits.
# Protocol is fixed: -ngl 99 -t 16 -c 512 -b 512 (see M9/M11 READMEs).
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m11
REFDIR=experiments/2026-08-28-m11-holdout/artifacts/reference-logits
LOGDIR=experiments/2026-08-28-m11-holdout/artifacts/logs

mkdir -p "$REFDIR" "$LOGDIR"

echo "== BF16 SHA-256 check =="
echo "8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc  $SRC" | sha256sum -c -

for d in wiki_test wiki_valid chinese code agent_chat; do
  echo "== reference: $d =="
  "$RT/llama-perplexity" \
    -m "$SRC" \
    -f "$HOLD/holdout-$d-64k.txt" \
    -ngl 99 -t 16 -c 512 -b 512 \
    --kl-divergence-base "$REFDIR/holdout-bf16-$d.kld" \
    > "$LOGDIR/ref-holdout-bf16-$d.log" 2>&1
  tail -2 "$LOGDIR/ref-holdout-bf16-$d.log"
done

echo "== kld headers =="
python3 - <<'EOF'
import struct, pathlib
for p in sorted(pathlib.Path("experiments/2026-08-28-m11-holdout/artifacts/reference-logits").glob("*.kld")):
    with p.open("rb") as f:
        magic = f.read(8)
        n_ctx, n_vocab, n_chunk = struct.unpack("iii", f.read(12))
    assert magic == b"_logits_" and n_ctx == 512 and n_vocab == 248320, p
    print(p.name, "n_ctx", n_ctx, "n_vocab", n_vocab, "n_chunk", n_chunk)
EOF
echo "STAGE A DONE"
