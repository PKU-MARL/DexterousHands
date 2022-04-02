#!/bin/bash

TASK=$1
ALGO=$2
NUM_ENVS=$3

echo "Experiments started."
for seed in $(seq 0 2)
do
    python train.py --task $TASK  --seed $seed   --algo=${ALGO} --num_envs=${NUM_ENVS}
done
echo "Experiments ended."
