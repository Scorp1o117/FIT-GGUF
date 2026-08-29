#!/usr/bin/env bash
# P1 regression replay: run the fit CLI end to end for the two frozen
# ground-truth points (Huihui FIT-50 O+B, Granite FIT-50 O+B).
set -euo pipefail
cd /run/media/s117/OS/FIT-GGUF
export PYTHONPATH=src

P1=experiments/2026-08-29-p1-cli
FITDIR=artifacts/fit/p1-replay
mkdir -p "$P1/huihui" "$P1/granite" "$FITDIR"

echo "== G1: Huihui analyze =="
python3 -m fit_gguf.cli analyze \
  --source artifacts/source/Huihui-Qwen3.8-27B-abliterated-BF16.gguf \
  --imatrix imatrix_unsloth.gguf \
  --runtime tools/llama-b10666-rocm \
  --out-dir "$P1/huihui/analysis"

echo "== G2/G3: Huihui plans =="
python3 -m fit_gguf.cli plan --analysis "$P1/huihui/analysis/analysis.json" \
  --fit 0.5 --policy original --out-prefix "$P1/huihui/o-fit50"
python3 -m fit_gguf.cli plan --analysis "$P1/huihui/analysis/analysis.json" \
  --fit 0.5 --policy balanced --out-prefix "$P1/huihui/b-fit50"

expect () { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['predicted_size_bytes'])" "$1"; }

echo "== G4: Huihui quantize =="
python3 -m fit_gguf.cli quantize --analysis "$P1/huihui/analysis/analysis.json" \
  --tensor-types "$P1/huihui/o-fit50-tensor-types.txt" \
  --out "$FITDIR/Huihui-O-FIT50-replay.gguf" \
  --expect-bytes "$(expect $P1/huihui/o-fit50-plan.json)"
python3 -m fit_gguf.cli quantize --analysis "$P1/huihui/analysis/analysis.json" \
  --tensor-types "$P1/huihui/b-fit50-tensor-types.txt" \
  --out "$FITDIR/Huihui-B-FIT50-replay.gguf" \
  --expect-bytes "$(expect $P1/huihui/b-fit50-plan.json)"

echo "== G5: Granite analyze =="
python3 -m fit_gguf.cli analyze \
  --source artifacts/source/granite-4.2-8b-BF16.gguf \
  --imatrix imatrix-granite-apex-c512.gguf \
  --runtime tools/llama-b10666-rocm \
  --out-dir "$P1/granite/analysis"

echo "== G6/G7: Granite plans =="
python3 -m fit_gguf.cli plan --analysis "$P1/granite/analysis/analysis.json" \
  --fit 0.5 --policy original --out-prefix "$P1/granite/o-fit50"
python3 -m fit_gguf.cli plan --analysis "$P1/granite/analysis/analysis.json" \
  --fit 0.5 --policy balanced --out-prefix "$P1/granite/b-fit50"

echo "== G8: Granite quantize =="
python3 -m fit_gguf.cli quantize --analysis "$P1/granite/analysis/analysis.json" \
  --tensor-types "$P1/granite/o-fit50-tensor-types.txt" \
  --out "$FITDIR/Granite-O-FIT50-replay.gguf" \
  --expect-bytes "$(expect $P1/granite/o-fit50-plan.json)"
python3 -m fit_gguf.cli quantize --analysis "$P1/granite/analysis/analysis.json" \
  --tensor-types "$P1/granite/b-fit50-tensor-types.txt" \
  --out "$FITDIR/Granite-B-FIT50-replay.gguf" \
  --expect-bytes "$(expect $P1/granite/b-fit50-plan.json)"

echo "== Gate evaluation =="
python3 scripts/evaluate_p1_replay.py
echo "STAGE P1-REPLAY DONE"
