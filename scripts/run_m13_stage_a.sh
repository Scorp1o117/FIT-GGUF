#!/usr/bin/env bash
# M13 stage A: verify BF16 provenance and generate five holdout-3 reference logits.
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF

RT=tools/llama-b10666-rocm
SRC=artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf
HOLD=/run/media/s117/OS/Models/eval-data/holdout-m13
REFDIR=experiments/2026-08-28-m13-budget-rule/artifacts/reference-logits
LOGDIR=experiments/2026-08-28-m13-budget-rule/artifacts/logs
mkdir -p "$REFDIR" "$LOGDIR"

echo "== BF16 SHA-256 check =="
echo "8a033407c8f58d43102aade25b973cc6d2f2ce5c5cbf4dc75a2cdb60b9e33cbc  $SRC" | sha256sum -c -

for d in wiki_test wiki_valid chinese code agent_chat; do
  echo "== reference: $d =="
  "$RT/llama-perplexity" \
    -m "$SRC" \
    -f "$HOLD/holdout3-$d-64k.txt" \
    -ngl 99 -t 16 -c 512 -b 512 \
    --kl-divergence-base "$REFDIR/holdout3-bf16-$d.kld" \
    > "$LOGDIR/ref-holdout3-bf16-$d.log" 2>&1
  tail -2 "$LOGDIR/ref-holdout3-bf16-$d.log"
done

echo "== artifact hash verification =="
FITDIR=artifacts/fit
sha256sum -c - <<EOF
d757266985bdbfe8a2df7a1d6f209effaf192036ece6be9b3391cb3c2dcef4e2  $FITDIR/Huihui-Qwen3.8-27B-FIT-25.gguf
e4fe1c46ab89c8b6343203168ebeec699372c2fb21f411c61c47edc2e1f33306  $FITDIR/Huihui-Qwen3.8-27B-FIT-50.gguf
4a05fccd1b0f77c51ff8d7f4be43663e85adc07f7f6ab1eece0ce5f14f00fad1  $FITDIR/Huihui-Qwen3.8-27B-FIT-75.gguf
191781ca4d12eaec7c0704828d695c722f302d011c028d6b813df1ffb1df35a7  $FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT25.gguf
7cfa1b91600115c046cb9afcae8347adc7de5a77b0d880b47a956cb8a4799a07  $FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT50.gguf
6023d58ab00876671af1eee9ece957fdea7b69d38541de88e7664824bd4f950e  $FITDIR/Huihui-Qwen3.8-27B-BLOCK-BALANCED-FIT75.gguf
EOF
echo "STAGE M13-A DONE"
