#!/bin/zsh
# Axial-chain duu (chains along the cylinder axis) at W=3, J3=0.40, chi=128, all D.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for D in 0.5 0.1 0.15 0.21 0.3 0.05 0.08 0.12 0.8 1.0 1.25 1.5; do echo $D; done |
  xargs -P 3 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 3 duu 128 axial >> dmrg_pd_J0.40_axial.jsonl 2>> logs/axial.err || echo "FAIL D=$0" >> logs/axial.err'
