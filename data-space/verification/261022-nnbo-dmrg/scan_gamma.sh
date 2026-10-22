#!/bin/zsh
# Gamma (+K) set: J3=0.40, W=2, chi=64, no Sz conservation; after the chi=256 small-D set.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_priority.sh" >/dev/null; do sleep 60; done
for P in u du duu duuu duuuu duuuuu; do echo "0.1 $P"; done |
  xargs -P 6 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 2 $1 64 >> dmrg_pd_J0.40_W2ref.jsonl 2>> logs/gamma.err'
xargs -P ${NP:-6} -L 1 zsh -c '.venv/bin/python dmrg_k.py 0.40 $0 $1 $2 2 $3 64 $4 >> dmrg_kg_J0.40_W2.jsonl 2>> logs/gamma.err || echo "FAIL D=$0 K=$1 h=$2 $3 G=$4" >> logs/gamma.err' < jobs_gamma.txt
