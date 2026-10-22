#!/bin/zsh
# W=4 ring (prod) vs axial orientation, J3=0.40, chi=256.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 4 $1 256 $2 >> dmrg_pd_J0.40_W4.jsonl 2>> logs/W4.err || echo "FAIL D=$0 $1 $2" >> logs/W4.err' < jobs_W4.txt
