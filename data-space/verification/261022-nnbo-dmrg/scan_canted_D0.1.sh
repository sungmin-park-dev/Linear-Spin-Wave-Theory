#!/bin/zsh
# Canted-phase check at D=0.1 (queued after scan_canted.sh): J3=0.40, W=3, Lx=12, chi=128, seed 1.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_canted.sh" >/dev/null; do sleep 60; done
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_sector.py 0.40 $0 3 12 $1 1 128 >> dmrg_sector_J0.40.jsonl 2>> logs/canted.err || echo "FAIL D=$0 nd=$1" >> logs/canted.err' < jobs_canted_D0.1.txt
