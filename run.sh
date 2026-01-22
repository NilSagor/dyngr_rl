#!/bin/bash
set -e

CONFIG="experiment_framework/configs/dygformer_config.yaml"
SCRIPT="experiment_framework/src/experiments/train.py"

for eval_type in transductive inductive; do
  for neg_sample in random historical inductive; do
    echo "Running: $eval_type + $neg_sample"
    python "$SCRIPT" \
      --configs "$CONFIG" \
      --override \
        data.evaluation_type=$eval_type \
        data.negative_sampling_strategy=$neg_sample
  done
done

# chmod +x run_all_dygformer.sh
# ./run_all_dygformer.sh