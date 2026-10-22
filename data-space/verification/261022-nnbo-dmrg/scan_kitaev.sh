#!/bin/zsh
# Kitaev set, stage A: J3=0.40, D=0.3, W=2, chi=64, no Sz conservation. Starts after the smallD scan.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_(smallD|priority|canted|fineM|W4|xxz|after|W6|devil|cchain)" >/dev/null; do sleep 60; done
# conserved W=2 chi=64 reference for the K=0 validation
for P in u du duu duuu duuuu duuuuu; do echo "0.3 $P"; done |
  xargs -P 5 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 2 $1 64 >> dmrg_pd_J0.40_W2ref.jsonl 2>> logs/kitaev.err'
xargs -P ${NP:-5} -L 1 zsh -c '.venv/bin/python dmrg_k.py 0.40 0.3 $0 $1 2 $2 64 >> dmrg_k_J0.40_D0.3_W2.jsonl 2>> logs/kitaev.err || echo "FAIL K=$0 h=$1 $2" >> logs/kitaev.err' < jobs_kitaev.txt
