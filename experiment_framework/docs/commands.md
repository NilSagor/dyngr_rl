# 🧪 Experiment Command Reference

### CLI patterns, overrides, sweep templates
> 💡 **Path Convention**: All commands assume you're running from the `haiyang_exp/` root directory.

## Quick Patterns
- Single seed: `--seeds 42`
- Multi-seed: `--seeds 42 123 456`
- Inductive: `--override data.evaluation_type=inductive`
- Ablation: `--override model.use_memory=false`

## Model-Specific Commands

# 🧪 Experiment Command Reference



## 🚀 1. Baseline & Single-Seed Runs
```bash
# GraphMixer - Transductive (default)
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/graphmixer.yaml \
  --seeds 42

# TAWRMAC - Transductive  
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42

# FreeDyG - Transductive
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/freedyg.yaml \
  --seeds 42


## Multi-Seed Runs (Reproducibility)

# Run with 3 seeds sequentially
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/graphmixer.yaml \
  --seeds 42 123 456

# Run with 5 seeds for robust statistics
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 123 456 789 101112

# Multi-seed with timestamped logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/graphmixer.yaml \
  --seeds 42 123 456 \
  --override logging.log_dir="experiment_framework/logs/graphmixer_multiseed_${TIMESTAMP}" \
  2>&1 | tee "experiment_framework/logs/graphmixer_multiseed_${TIMESTAMP}.log"


# Explicitly set transductive (usually default)
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override data.evaluation_type=transductive \
             data.negative_sampling_strategy=random

# Switch to inductive mode
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override data.evaluation_type=inductive \
             data.unseen_ratio=0.1 \
             data.negative_sampling_strategy=inductive

# GraphMixer inductive
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/graphmixer.yaml \
  --seeds 42 \
  --override data.evaluation_type=inductive \
             data.unseen_ratio=0.15 \
             data.negative_sampling_strategy=inductive


# 4. Cross-Dataset Evaluation

for DATASET in wikipedia reddit mooc lastfm; do
  python experiment_framework/src/experiments/train_v5.py \
    --config experiment_framework/configs/baseline/graphmixer.yaml \
    --seeds 42 \
    --override data.dataset="${DATASET}" \
               experiment.name="graphmixer_${DATASET}_transductive"
done

# 5. Ablation Studies - TAWRMAC Component Removal

# Disable memory module
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override model.use_memory=false \
             experiment.name="tawrmac_abl_no_memory"

# Disable walk module  
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override model.enable_walk=false \
             experiment.name="tawrmac_abl_no_walk"

# Disable co-occurrence
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override model.enable_neighbor_cooc=false \
             experiment.name="tawrmac_abl_no_cooc"

# Disable restart mechanism
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override model.enable_restart=false \
             experiment.name="tawrmac_abl_no_restart"

# Full Ablation 
for MEMORY in true false; do
  for WALK in true false; do
    for COOC in true false; do
      NAME="abl_mem${MEMORY}_walk${WALK}_cooc${COOC}"
      python experiment_framework/src/experiments/train_v5.py \
        --config experiment_framework/configs/baseline/tawrmac_config.yaml \
        --seeds 42 \
        --override model.use_memory=${MEMORY} \
                   model.enable_walk=${WALK} \
                   model.enable_neighbor_cooc=${COOC} \
                   experiment.name="tawrmac_${NAME}" \
                   training.max_epochs=10  # Shorter for ablation
    done
  done
done

# HiCoST Variant Ablations
# Disable explicit Co-GNN in HiCoSTdev1
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/hicost_dev/v1.yaml \
  --seeds 42 \
  --override model.use_explicit_co_gnn=false \
             experiment.name="hicostdev1_abl_no_co_gnn"

# Disable time-delta attention in HiCoSTdev2
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/hicost_dev/v2.yaml \
  --seeds 42 \
  --override model.use_time_delta_attention=false \
             experiment.name="hicostdev2_abl_no_time_delta"

# 6. Parameter Sweeps
# Sweep learning rates (GraphMixer)
for LR in 1e-4 3e-4 1e-3 3e-3; do
  python experiment_framework/src/experiments/train_v5.py \
    --config experiment_framework/configs/baseline/graphmixer.yaml \
    --seeds 42 123 456 \
    --override training.learning_rate=${LR} \
               experiment.name="graphmixer_lr${LR}"
done

# Sweep memory dimensions (TAWRMAC)
for MEM_DIM in 64 128 172 256; do
  python experiment_framework/src/experiments/train_v5.py \
    --config experiment_framework/configs/baseline/tawrmac_config.yaml \
    --seeds 42 \
    --override model.memory_dim=${MEM_DIM} \
               model.time_dim=${MEM_DIM} \
               model.walk_emb_dim=${MEM_DIM} \
               experiment.name="tawrmac_memdim${MEM_DIM}"
done

# Sweep walk hyperparameters
for WALK_LEN in 2 4 6 8; do
  for NUM_WALKS in 5 10 20; do
    python experiment_framework/src/experiments/train_v5.py \
      --config experiment_framework/configs/baseline/tawrmac_config.yaml \
      --seeds 42 \
      --override model.walk_length=${WALK_LEN} \
                 model.num_walks=${NUM_WALKS} \
                 experiment.name="tawrmac_walk${WALK_LEN}x${NUM_WALKS}" \
                 training.max_epochs=20  # Shorter for sweep
  done
done

# 7. Sensitivity & Advanced Runners
# Run pre-defined sensitivity study
python experiment_framework/src/experiments/main_sensitivityV2.py \
  --config experiment_framework/configs/sensitivity_config.yaml \
  --study walk_distribution \
  --seeds 42 43 44

# Filter specific configs + top-k selection
python experiment_framework/src/experiments/main_sensitivityV2.py \
  --config experiment_framework/configs/sensitivity_config.yaml \
  --study memory_dim \
  --filter 64 128 172 \
  --top-k 3 \
  --seeds 42 123 456

# Auto-select best configs by validation AP
python experiment_framework/src/experiments/main_sensitivityV2.py \
  --config experiment_framework/configs/sensitivity_config.yaml \
  --study walk_distribution \
  --top-k 2 \
  --metric val_ap \
  --mode max \
  --seeds 42 123 456 789

# 8. Debugging & Profiling
# Minimal run: 1 epoch, batch=8, debug logging
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/graphmixer.yaml \
  --seeds 42 \
  --override training.max_epochs=1 \
             training.batch_size=8 \
             experiment.debug=true \
             hardware.num_workers=0

# Profile FLOPs only (skip training)
python experiment_framework/src/experiments/train_v5.py \
  --config experiment_framework/configs/baseline/tawrmac_config.yaml \
  --seeds 42 \
  --override experiment.debug=true \
             profiling.enabled=true \
             training.max_epochs=0