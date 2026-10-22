#!/bin/zsh
# Canted-phase check: J3=0.40, W=3, Lx=12 (72-site cell), chi=128, random start (seed 1), Sz conserved.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P ${NP:-4} -L 1 zsh -c '.venv/bin/python dmrg_sector.py 0.40 $0 3 12 $1 1 128 >> dmrg_sector_J0.40.jsonl 2>> logs/canted.err || echo "FAIL D=$0 nd=$1" >> logs/canted.err' < jobs_canted.txt
