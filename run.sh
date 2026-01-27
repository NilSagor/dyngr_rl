#!/bin/bash
set -e

CONFIG="experiment_framework/configs/dygformer_config.yaml"
SCRIPT="experiment_framework/src/experiments/train.py"

DATASETS=("wikipedia" "reddit" "mooc" "lastfm" "uci")
EVAL_TYPES=("transductive" "inductive")
SEEDS=(42 43 44)

for dataset in "${DATASETS[@]}"; do
    for eval_type in transductive inductive; do
        for neg_sample in random historical inductive; do
            for seed in "${SEEDS[@]}"; do
                # echo "Running: $eval_type + $neg_sample"
                echo "Running: $dataset | $eval_type | $neg_strat | seed=$seed"
                python "$SCRIPT" \
                --configs "$CONFIG" \
                --override \
                    data.evaluation_type=$eval_type \
                    data.negative_sampling_strategy=$neg_sample
            done
        done
    done
done


# #!/bin/bash
# set -e

# CONFIG="experiment_framework/configs/dygformer_config.yaml"
# SCRIPT="experiment_framework/src/experiments/train.py"

# MODELS=("DyGFormer" "TGN" "TAWRMAC")
# DATASETS=("wikipedia" "reddit" "mooc" "lastfm" "uci")
# EVAL_TYPES=("transductive" "inductive")
# NEG_STRATS=("random" "historical" "inductive")
# SEEDS=(42 43 44)

# for model in "${MODELS[@]}"; do
#   for dataset in "${DATASETS[@]}"; do
#     for eval_type in "${EVAL_TYPES[@]}"; do
#       for neg_strat in "${NEG_STRATS[@]}"; do
#         for seed in "${SEEDS[@]}"; do
#           echo "Running: $model | $dataset | $eval_type | $neg_strat | seed=$seed"
#           python "$SCRIPT" \
#             --configs "$CONFIG" \
#             --override \
#               model.name="$model" \
#               data.dataset="$dataset" \
#               data.evaluation_type="$eval_type" \
#               data.negative_sampling_strategy="$neg_strat" \
#               experiment.seed="$seed"
#         done
#       done
#     done
#   done
# done


# MODELS=("DyGFormer" "TGN" "TAWRMAC")
# DATASETS=("wikipedia" "reddit" "mooc" "lastfm" "uci")
# EVAL_TYPES=("transductive" "inductive")
# NEG_STRATS=("random" "historical" "inductive")
# SEEDS=(42 43 44)

# for model in "${MODELS[@]}"; do
#   for dataset in "${DATASETS[@]}"; do
#     for eval_type in "${EVAL_TYPES[@]}"; do
#       for neg_strat in "${NEG_STRATS[@]}"; do
#         for seed in "${SEEDS[@]}"; do
#           echo "Running: $model | $dataset | $eval_type | $neg_strat | seed=$seed"
#           python "$SCRIPT" \
#             --configs "$CONFIG" \
#             --override \
#               model.name="$model" \
#               data.dataset="$dataset" \
#               data.evaluation_type="$eval_type" \
#               data.negative_sampling_strategy="$neg_strat" \
#               experiment.seed="$seed"
#         done
#       done
#     done
#   done
# done







# chmod +x run_all_dygformer.sh
# ./run_all_dygformer.sh