#!/bin/zsh
# Fine M(h): uniform-start sector scans, J3=0.40, W=3, chi=128. Columns: D Lx n_down seed.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_sector.py 0.40 $0 3 $1 $2 $3 128 >> dmrg_sector_J0.40.jsonl 2>> logs/canted.err || echo "FAIL D=$0 Lx=$1 nd=$2 seed=$3" >> logs/canted.err' < jobs_fineM.txt
