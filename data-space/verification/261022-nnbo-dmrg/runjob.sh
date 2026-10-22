#!/bin/zsh
# usage: runjob.sh W pattern chi [fallback_chi]
cd ~/nnbo-dmrg
export OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
W=$1; P=$2; C=$3; F=$4
out=results_W${W}_${P}_chi${C}.txt
/usr/bin/time -l .venv/bin/python dmrg_standalone.py 0.49 1.25 $W $P 0.47 $C > $out 2> logs/W${W}_${P}_chi${C}.err
rc=$?
echo "rc=$rc $(date)" >> logs/W${W}_${P}_chi${C}.err
if [[ $rc -ne 0 && -n "$F" ]]; then
  exec $0 $W $P $F
fi
