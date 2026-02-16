#!/bin/bash
set -e

CONFIG="experiment_framework/configs/tgn_config.yaml"
SCRIPT="experiment_framework/src/experiments/train_v2.py"  # Use fixed train_v2.py

DATASETS=("wikipedia" "Contacts" "CanParl" "Flights" "myket" "UNtrade" "USLegis" "reddit" "mooc" "lastfm")
SEEDS=(42 43 44)

# VALID COMBINATIONS PER EVALUATION TYPE (TGN ICML 2020 compliant)
declare -A VALID_STRATEGIES
VALID_STRATEGIES["transductive"]="random historical"
VALID_STRATEGIES["inductive"]="random inductive"  # Historical invalid for unseen nodes

for dataset in "${DATASETS[@]}"; do
    for eval_type in "${!VALID_STRATEGIES[@]}"; do
        strategies="${VALID_STRATEGIES[$eval_type]}"
        
        for neg_sample in $strategies; do
            for seed in "${SEEDS[@]}"; do
                echo "=================================================================="
                echo "Running: dataset=$dataset | eval=$eval_type | sampling=$neg_sample | seed=$seed"
                echo "=================================================================="
                
                # Skip invalid combinations BEFORE launching expensive training
                if [[ "$eval_type" == "transductive" && "$neg_sample" == "inductive" ]]; then
                    echo "❌ SKIPPED: Inductive sampling invalid for transductive evaluation"
                    continue
                fi
                
                if [[ "$eval_type" == "inductive" && "$neg_sample" == "historical" ]]; then
                    echo "⚠️  WARNING: Historical sampling suboptimal for inductive evaluation (unseen nodes have no history)"
                    # Still run but with warning (for ablation studies only)
                fi
                
                # LAUNCH TRAINING WITH VALID CONFIG
                python "$SCRIPT" \
                    --config "$CONFIG" \
                    --override \
                        data.dataset="$dataset" \
                        data.evaluation_type="$eval_type" \
                        data.negative_sampling_strategy="$neg_sample" \
                        experiment.seed="$seed" \
                || {
                    echo "❌ FAILED: dataset=$dataset eval=$eval_type sampling=$neg_sample seed=$seed"
                    exit 1
                }
                
                echo "✅ COMPLETED: dataset=$dataset eval=$eval_type sampling=$neg_sample seed=$seed"
                echo ""
            done
        done
    done
done

echo "=================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "=================================================================="`