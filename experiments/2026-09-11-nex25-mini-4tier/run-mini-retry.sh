#!/bin/bash
set -uo pipefail
cd /run/media/s117/OS/FIT-GGUF
# Lower preset (IQ3_XXS, 12.69 GiB @ kld 0.1860) already PASSES the mini gate, so the
# search bisects DOWN toward the smallest passing size instead of walking up from a
# failing anchor — the failure mode that left the first attempt budget_exhausted.
PYTHONPATH=/run/media/s117/OS/FIT-GGUF/src /home/s117/heretic-env/bin/python -c "
import sys
from fit_gguf.cli import main
sys.argv = ['fit','fidelity-search',
  '--source','artifacts/source/Nex-N2.5-mini-abliterix-v8-t17-BF16.gguf',
  '--imatrix','experiments/2026-09-11-nex25-mini-4tier/calibration/calibration-imatrix.gguf',
  '--runtime','tools/llama-b10666-rocm',
  '--refs-dir','experiments/2026-09-11-nex25-mini-4tier/calibration/references',
  '--eval-data-dir','eval-data',
  '--tier','mini',
  '--preset-ladder','IQ3_XXS,Q4_K_M',
  '--manifest','experiments/2026-09-11-nex25-mini-4tier/calibration/state-artifact-manifest.txt',
  '--logs-dir','/home/s117/fit-logs/nex25-fs/mini2',
  '--out-dir','experiments/2026-09-11-nex25-mini-4tier/fs/mini2',
  '--work-dir','/home/s117/fit-scratch/fs-mini2',
  '--profile','precise',
  '--threads','16',
  '--n-gpu-layers','30']
main()
" >> /home/s117/fit-logs/nex25-fs/mini2/driver.log 2>&1
echo "END mini2 rc=$? $(date '+%F %T')" >> /home/s117/fit-logs/nex25-fs/status.txt
