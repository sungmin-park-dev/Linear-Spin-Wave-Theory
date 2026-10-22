#!/bin/zsh
# XXZ test: J3=0.40, W=3, chi=128, J1^zz = -1 + dJ1, J1^xy = -1.  Columns: D pattern init dJ1.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 3 $1 128 $2 $3 >> dmrg_xxz_J0.40.jsonl 2>> logs/xxz.err || echo "FAIL D=$0 $1 $2 dJ1=$3" >> logs/xxz.err' < jobs_xxz.txt
