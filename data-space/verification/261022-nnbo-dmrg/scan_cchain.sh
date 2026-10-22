#!/bin/zsh
# Test of the neutron-type structure (o up o down up up): chains 'o' seeded with Sz=0 / half-polarized.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
xargs -P 3 -L 1 zsh -c '.venv/bin/python dmrg_pd.py 0.40 $0 $1 $2 $3 >> dmrg_pd_J0.40_cchain.jsonl 2>> logs/cchain.err || echo "FAIL D=$0 W=$1 $2" >> logs/cchain.err' < jobs_canted_chain.txt
