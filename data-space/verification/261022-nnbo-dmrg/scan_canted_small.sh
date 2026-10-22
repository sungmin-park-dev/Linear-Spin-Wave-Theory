#!/bin/zsh
# Lx=12 sector scan with 'uniform' (up/0, no down chains) starts, after scan_canted.sh:
# J3=0.40, W=3, chi=128; D=0.1, 0.12 (M=1/12..11/12) and D=0.21, 0.5 (M=1/3..11/12).
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_canted.sh" >/dev/null; do sleep 60; done
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_sector.py 0.40 $0 3 12 $1 $2 128 >> dmrg_sector_J0.40.jsonl 2>> logs/canted.err || echo "FAIL D=$0 nd=$1 seed=$2" >> logs/canted.err' < jobs_canted_small.txt
