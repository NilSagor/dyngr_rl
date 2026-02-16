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

# PHASE 1: Transductive baseline (2 strategies × 10 datasets × 3 seeds = 60 runs)
<!-- ./run_experiments.sh transductive -->
./run_v2.sh transductive
# PHASE 2: Inductive evaluation (2 strategies × 10 datasets × 3 seeds = 60 runs)
./run_experiments.sh inductive

# PHASE 3: Ablation studies (historical in inductive - optional)
./run_experiments.sh inductive_ablation