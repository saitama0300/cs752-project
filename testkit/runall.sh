#!/bin/bash
set -x
set -e

CMD="python model.py"

# for i in 1 4 16 64 256 1024 ; do
#    APP=nerf-pytorch BS=$i NI=30 ./../utils/run_bench.sh $CMD
#    APP=nerf-pytorch BS=$i NI=30 ./../utils/run_prof.sh $CMD
#    APP=nerf-pytorch BS=$i NI=30 ./../utils/run_nsys.sh $CMD
# done

for i in 1 1024 ; do
    APP=test_model BS=$i NI=30 ./../utils/run_ncu.sh $CMD
done
