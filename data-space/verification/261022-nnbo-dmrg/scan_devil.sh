#!/bin/zsh
# Devil's-staircase test: mixed chain arrangements between M=0 and 1/2 (ring orientation), W=3, chi=128.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 zsh -c '.venv/bin/python dmrg_pd.py $0 $1 3 $2 128 $3 >> dmrg_pd_devil.jsonl 2>> logs/devil.err || echo "FAIL $0 $1 $2 $3" >> logs/devil.err' < jobs_devil.txt
