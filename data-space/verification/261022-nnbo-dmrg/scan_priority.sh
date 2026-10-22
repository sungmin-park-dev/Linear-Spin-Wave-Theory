#!/bin/zsh
# Small-D refinement (after smallD): pattern runs, chi=256 for D<=0.12.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_smallD.sh" >/dev/null; do sleep 60; done
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_pd.py $0 $1 3 $2 $3 >> dmrg_pd_smallD2.jsonl 2>> logs/priority.err || echo "FAIL $0 $1 $2 $3" >> logs/priority.err' < jobs_priority.txt
