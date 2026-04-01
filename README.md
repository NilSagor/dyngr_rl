# Dynamic Graph Representation Learning

<!-- python experiment_framework/src/experiments/train.py --config experiment_framework/configs/dygformer_config.yaml -->

Quick Debug run

```bash
    python experiment_framework/src/experiments/train.py \
    --configs experiment_framework/configs/dygformer_config.yaml \
    --override \
        model.name="DyGFormer" \
        data.dataset="wikipedia" \
        data.evaluation_type="transductive" \
        data.negative_sampling_strategy="random" \
        experiment.seed=42

```
<!-- python experiment_framework/src/experiments/train.py --configs experiment_framework/configs/dygformer_config.yaml --override model.name="DyGFormer" data.dataset="wikipedia" data.valuation_type="transductive" data.negative_sampling_strategy="random" experiment.seed=42 -->


# Run only specific configs from walk_distribution
python main_sensitivity.py --config configs/sensitivity_config.yaml --study walk_distribution --seeds 42 43 44 --filter balanced tawr_heavy

# Run only the short_heavy config
python main_sensitivity.py --config configs/sensitivity_config.yaml --study walk_distribution --seeds 42 --filter short_heavy

# Run all configs (no filter)
python main_sensitivity.py --config configs/sensitivity_config.yaml --study walk_distribution --seeds 42 43 44

# Filter works with partial matches (case-insensitive)
python main_sensitivity.py --config configs/sensitivity_config.yaml --study walk_distribution --filter balanced  # Matches "balanced_v2" too

# Auto-select top 2 configs (will pick: tawr_heavy, balanced)
python main_sensitivity.py --config configs/sensitivity_config.yaml \
  --study walk_distribution \
  --top-k 2 \
  --seeds 42 43 44 45 46

# Or manually specify
python main_sensitivity.py --config configs/sensitivity_config.yaml \
  --study walk_distribution \
  --filter balanced tawr_heavy \
  --seeds 42 43 44 45 46


# PHASE 1: Transductive baseline (2 strategies × 10 datasets × 3 seeds = 60 runs)
<!-- ./run_experiments.sh transductive -->
./run_v2.sh transductive
# PHASE 2: Inductive evaluation (2 strategies × 10 datasets × 3 seeds = 60 runs)
./run_experiments.sh inductive

# PHASE 3: Ablation studies (historical in inductive - optional)
./run_experiments.sh inductive_ablation