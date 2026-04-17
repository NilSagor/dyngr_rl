# HiCoST: Hierarchical Co-occurrence Spatio-Temporal Network

<!-- python experiment_framework/src/experiments/train.py --config experiment_framework/configs/dygformer_config.yaml -->

## Run HiCoST V3 training
python experiment_framework/src/experiments/train_v4.py --config experiment_framework/configs/hicost_configv2.yaml  --seeds 42

## TAWRMAC training
python experiment_framework/src/experiments/train_v5.py \
    --config experiment_framework/configs/tawrmac_config.yaml \
    --seeds 42

### multiple seed
python src/experiments/train_v5.py \
    --config configs/tawrmac_config.yaml \
    --seeds 42 123 456


### Quick Debug run

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


# Run only the short_heavy config


# Run all configs (no filter)
python experiment_framework/src/experiments/main_sensitivityV2.py --config experiment_framework/configs/sensitivity_config.yaml --study walk_distribution --seeds 42 43 44

# Filter works with partial matches (case-insensitive)
python experiment_framework/src/experiments/main_sensitivityV2.py --config experiment_framework/configs/sensitivity_config.yaml --study walk_distribution --filter balanced  # Matches "balanced_v2" too

# Auto-select top 2 configs (will pick: tawr_heavy, balanced)
python experiment_framework/src/experiments/main_sensitivityV2.py --config experiment_framework/configs/sensitivity_config.yaml \
  --study walk_distribution \
  --top-k 2 \
  --seeds 42 43 44 45 46

# Or manually specify
python experiment_framework/src/experiments/main_sensitivityV2.py --config experiment_framework/configs/sensitivity_config.yaml \
  --study walk_distribution \
  --filter balanced tawr_heavy \
  --seeds 42 43 44 45 46

python experiment_framework/src/experiments/main_sensitivityV2.py --config experiment_framework/configs/sensitivity_config.yaml --study memory_dim   --seeds 42 --filter 64


# PHASE 1: Transductive baseline (2 strategies × 10 datasets × 3 seeds = 60 runs)
<!-- ./run_experiments.sh transductive -->
./run_v2.sh transductive
# PHASE 2: Inductive evaluation (2 strategies × 10 datasets × 3 seeds = 60 runs)
./run_experiments.sh inductive

# PHASE 3: Ablation studies (historical in inductive - optional)
./run_experiments.sh inductive_ablation