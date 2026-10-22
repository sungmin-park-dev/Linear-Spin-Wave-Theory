#!/bin/zsh
# W=6 axial (du, duu, duuuuu all fit) vs ring, J3=0.40, chi=256, D=0.3 and 1.25.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 zsh -c '/usr/bin/time -l .venv/bin/python dmrg_pd.py 0.40 $0 6 $1 256 $2 >> dmrg_pd_J0.40_W6.jsonl 2>> logs/W6_$0_$1_$2.err || echo "FAIL D=$0 $1 $2" >> logs/W6.err' < jobs_W6.txt
