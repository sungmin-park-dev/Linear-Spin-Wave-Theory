#!/bin/zsh
# usage: scan_pd.sh J3 outfile D1 D2 ...   (W=3, chi=128, 4 parallel, OMP=1)
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
J3=$1; OUT=$2; shift 2
for D in "$@"; do for P in u du duu duuu duuuu duuuuu; do echo "$D $P"; done; done |
  xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_pd.py '$J3' $0 3 $1 128 >> '$OUT' 2>> logs/pd.err || echo "FAIL J3='$J3' D=$0 $1" >> logs/pd.err'
