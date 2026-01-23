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
