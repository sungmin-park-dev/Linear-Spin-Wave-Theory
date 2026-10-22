#!/bin/zsh
# Rerun of the random-start (rand1) small-D jobs; the first pass ignored the init argument.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 2 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 3 $1 128 $2 >> dmrg_pd_J0.40_rand.jsonl 2>> logs/rand.err || echo "FAIL D=$0 $1 $2" >> logs/rand.err' < jobs_rand.txt
