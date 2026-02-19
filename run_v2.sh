#!/bin/bash
set -e

CONFIG="experiment_framework/configs/tgn_config.yaml"
SCRIPT="experiment_framework/src/experiments/train_v2.py"

DATASETS=("wikipedia" "Contacts" "CanParl" "Flights" "myket" "UNtrade" "USLegis" "reddit" "mooc" "lastfm" "uci" "enron")
SEEDS=(42 43 44)

# If argument provided, restrict to that eval type; otherwise run all
if [ -n "$1" ]; then
    REQUESTED_EVAL="$1"
    if [[ "$REQUESTED_EVAL" != "transductive" && "$REQUESTED_EVAL" != "inductive" && "$REQUESTED_EVAL" != "inductive_ablation" ]]; then
        echo "Invalid argument: use 'transductive', 'inductive', or 'inductive_ablation'"
        exit 1
    fi
else
    REQUESTED_EVAL="all"
fi

# Valid combinations per evaluation type
declare -A VALID_STRATEGIES
VALID_STRATEGIES["transductive"]="random historical"
VALID_STRATEGIES["inductive"]="random inductive"
VALID_STRATEGIES["inductive_ablation"]="random historical inductive"

# Determine which eval types to run
if [ "$REQUESTED_EVAL" == "all" ]; then
    EVAL_TYPES=("transductive" "inductive")
else
    EVAL_TYPES=("$REQUESTED_EVAL")
fi

for eval_type in "${EVAL_TYPES[@]}"; do
    # Map ablation to inductive for config
    if [ "$eval_type" == "inductive_ablation" ]; then
        strategies="${VALID_STRATEGIES[$eval_type]}"
        config_eval_type="inductive"
    else
        strategies="${VALID_STRATEGIES[$eval_type]}"
        config_eval_type="$eval_type"
    fi

    for neg_sample in $strategies; do
        for dataset in "${DATASETS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "=================================================================="
                echo "Running: dataset=$dataset | eval=$config_eval_type | sampling=$neg_sample | seed=$seed"
                echo "=================================================================="

                # Skip invalid combinations
                if [[ "$config_eval_type" == "transductive" && "$neg_sample" == "inductive" ]]; then
                    echo "❌ SKIPPED: Inductive sampling invalid for transductive evaluation"
                    continue
                fi

                if [[ "$config_eval_type" == "inductive" && "$neg_sample" == "historical" && "$eval_type" != "inductive_ablation" ]]; then
                    echo "⚠️  WARNING: Historical sampling suboptimal for inductive evaluation (unseen nodes have no history)"
                fi

                python "$SCRIPT" \
                    --config "$CONFIG" \
                    --override \
                        data.dataset="$dataset" \
                        data.evaluation_type="$config_eval_type" \
                        data.negative_sampling_strategy="$neg_sample" \
                        experiment.seed="$seed" \
                || {
                    echo "❌ FAILED: dataset=$dataset eval=$config_eval_type sampling=$neg_sample seed=$seed"
                    exit 1
                }

                echo "✅ COMPLETED: dataset=$dataset eval=$config_eval_type sampling=$neg_sample seed=$seed"
                echo ""
            done
        done
    done
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "=================================================================="