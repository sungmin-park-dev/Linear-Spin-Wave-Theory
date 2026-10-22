#!/bin/zsh
# Boundary refinement D_c(J3): J3=0.40/0.42/0.43 x D=0.13/0.17 (ring + rand1 + axial), W=3, chi=128.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 zsh -c '.venv/bin/python dmrg_pd.py $0 $1 3 $2 128 $3 >> dmrg_pd_J3bnd.jsonl 2>> logs/bnd.err || echo "FAIL $0 $1 $2 $3" >> logs/bnd.err' < jobs_bnd.txt
