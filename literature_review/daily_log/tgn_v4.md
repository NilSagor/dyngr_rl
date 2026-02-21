# Model Building Process 4

## Component: TGNv4 (Main Model)
- version: v4
- Date: 2026-02-20
- Status: Training Complete

### Purpose
The main orchestrator for temporal link prediction. It replaces the standard TGN memory updater with Stability-Augmented Memory (SAM) and adds Multi-Scale Walks to capture structural context beyond immediate neighbors.

### Architecture Overview
- Input: Source/Destination Nonde, Timestamps, Edge features
- Output: Link probability scores
- Key Mechanism:
    1. SAM updates node memory using prototypes instead of GRU
    2. Walks (short, long, TAWR) are sampled and encoded.
    3. Walk embeddings are fused with memory embeddings.

### Key Design Decisions:
1. Disabled standard memory_updater (set to None)
    - SAM module handles memory updates exclusively to improve stability.
2. Added fusion layer (linar + layernorm)
    - To combine walk embeddings (structural) with memory embeddings (temporal) without dimension mismatch.
3. used torch.autograd.set_detect_anomaly(True)
    - to Catch NaN/Inf gradients early during developement.


### Config & Hyperparameter
|Parameter|Value|Description|
|--:      |-:   |-:         |
|hidden_dim|172    ||
|num_prototypes  |5  ||
|walk_length_short|3||
|learning_rate|13-4||
|walk_length_long|10||
|-:        |-:|-:|

### Testing & Obs
- Training Time: 1588.7s for 40 epochs
- Best Validation AP: 0.80739 (Epoch 30)
- Test AP: 0.7745
- Issue: val_ap plateaued after Epoch 30 (EarlyStopping triggered).
- Hardware: NVidia GeForce RTX 4060Ti (CUDA used).

## Component: MultiScaleWalkerSampler
- version: v1.0
- Status:Active

### Purpose:
Generates temporal walks starting from source and target nodes. It creates three types of walks to capture different scales of graph context: Short (local), long (global) and TAWR (with learnblbe restarts).

### Architecture Overview
- uses `neighbor_cache` for fast lookup.
- Samples neighbors based on temporal bias (recent neighbors peferred).
- TAWR walks use a learnable restart probability based on node memory.

### key Desing
1. Implemented neighbor_cache as python dict.
    - Faster than querying edges index repeatedly during sampling.
    - Needs manual update when graph structure changes (`update_neighbors`)
2. Anonymized walks (`anonymize_walks`).
    - To prevent the model from memorizing specific node IDs during walk encoding.
3. Temporal bias Sampling
    - to ensure walks respect time direction (no future edges).

### Config & Hyperparameters

|Parameter|Value|Description|
|--:      |-:   |-:         |
|temperature|0.1    |Control sharpness of temporal bias|
|num_walks_short  |10  |walks per node for short scale|
|walk_length_tawr|8|Length for Temporal Anonymous walks|
|-:        |-:|-:|

### Testing & obs.
- Performance: Sampling is CPU bound (sample_short_walks)
- Optimization: Consider moving sampling to GPU if bottleneck occurs.
- Log Note: walks are generated in compute_temporal_embeddings_with_walks


## Component: WalkEncoder
version: v1.0
status: Active

### Purpose

Converts raw walk sequences (nodes + times) into fixed-size embedding vectors that can be fused with the main TGN Embeddings

### Architecture Overview:

- Input: Walk tensor (nodes, times, masks) from Sampler.

- Output: [batch_size, output_dim] embedding tensor.

- Key Mechansim: Hierarchical Attention

    1. Step Attention: Aggregates steps within a single walk.
    2. Walk Attention: Aggregates multiple walks of the same type.
    3. Scale Fusion: Aggregates short, long and TAWR representations.

### Key Design Decisions
1. Used `MultiheadAttention` instead of LSTM
    - Reason: Better at capturing long-range dependencies within walks.
2. Added `TimeEncoder` inside walk encoding.
    - Reason: Walk steps have timestamps; purely structural info is insufficient.
3. Mean pooling after attention.
    - Simple and effective aggregation for variable length walks.

### Config & Hyperparameters

|Parameter|Value|Description|
|--:      |-:   |-:         |
|num_heads|4    |Attention heads|
|dropout  |0.1  |Regularization|
|output_dim|172|Must match TGN hidden dim|
|-:        |-:|-:|

### Testing & Observations
- Integration: Fused via `fusion_layer` in TGNv4.
- Ablation: `use_walk_encoder` flag allows turning this off to test baseline TGN performance.

## Component Training Pipeline and Metrics

Version: V2(trainv_v2.py)
Status: Production

### Purpose
Manages the training loop, validation checks, checkpointing, and final evaluation.

### Architecture Overview:
- Framework: Lightning(Trainer)
- Dataset: Wikipeida (Transductive setting)
- Metrics: Accuracy, AP (Average Precision), AUC, Loss

### Testing & Observations
- Dataset name: Wikipedia
- Dataset Stats: 9228 nodes, 157474 edges
- Progression: 
    - Epoch 0:  val_ap: 0.760
    - Epoch 30: val_ap: 0.807 (best)
    - Epoch 40: Stopped (No imporovement)
- Final Test: AP: 0.74, AUC=0.869


