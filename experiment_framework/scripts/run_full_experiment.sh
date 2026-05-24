#!/bin/bash
# chmod +x scripts/run_full_experiment.sh
MODEL=${1:-graphmixer}
EVAL_TYPE=${2:-transductive}
SEEDS=${3:-"42 123 456"}
DATASET=${4:-wikipedia}

CONFIG="experiment_framework/configs/baseline/${MODEL}.yaml"
if [[ "${MODEL}" == "tawrmac" ]]; then
  CONFIG="experiment_framework/configs/baseline/tawrmac_config.yaml"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="${MODEL}_${DATASET}_${EVAL_TYPE}_${TIMESTAMP}"

echo "🚀 Running: ${EXP_NAME}"
echo "   Config: ${CONFIG}"
echo "   Seeds: ${SEEDS}"
echo "   Eval: ${EVAL_TYPE}"

python experiment_framework/src/experiments/train_v5.py \
  --config "${CONFIG}" \
  --seeds ${SEEDS} \
  --override data.dataset="${DATASET}" \
             data.evaluation_type="${EVAL_TYPE}" \
             data.negative_sampling_strategy="${EVAL_TYPE}" \
             experiment.name="${EXP_NAME}" \
             logging.log_dir="experiment_framework/outputs/${EXP_NAME}" \
  2>&1 | tee "experiment_framework/outputs/${EXP_NAME}/run.log"

echo "✅ Completed: ${EXP_NAME}"