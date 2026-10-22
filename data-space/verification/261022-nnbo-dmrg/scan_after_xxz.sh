#!/bin/zsh
# After the XXZ test: (a) boundary trace J3=0.42/0.43 x D=0.12/0.15/0.2 (ring+rand1+axial) and
# rand1 at J3=0.37/0.45 (D=0.1/0.15); then (b) Lx=12 uniform-start sectors at J3=0.45, D=0.15. W=3.
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "[s]can_xxz.sh" >/dev/null; do sleep 60; done
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_pd.py $0 $1 3 $2 128 $3 >> dmrg_pd_J3scan.jsonl 2>> logs/J3scan.err || echo "FAIL $0 $1 $2 $3" >> logs/J3scan.err' < jobs_J3scan.txt
touch logs/J3scan.done
xargs -P 4 -L 1 zsh -c '.venv/bin/python dmrg_sector.py $0 $1 3 $2 $3 $4 128 >> dmrg_sector_J0.45.jsonl 2>> logs/canted.err || echo "FAIL J3=$0 D=$1 nd=$3" >> logs/canted.err' < jobs_fineM_J045.txt
