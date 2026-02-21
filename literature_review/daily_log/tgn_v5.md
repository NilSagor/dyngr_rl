version: v5
Date: 2026-02-21
Status: Training Complete


Architecture Overview
Input: Source/Destination Nonde, Timestamps, Edge features
Output: Link probability scores

Key Mechanism:
1. SAM updates node memory using prototypes instead of GRU
2. Walks (short, long, TAWR) are sampled and encoded.
3. Walk embeddings are fused with memory embeddings.

Key Design Decisions:




Component: WalkEncoder
version: v1.0
status: Active

### Purpose

Converts raw walk sequences (nodes + times) into fixed-size embedding vectors that can be fused with the main TGN Embeddings

Architecture Overview:

input: Walk tensor (nodes, times, masks) from Sampler.

Output: [batch_size, output_dim] embedding tensor.

Key Mechansim: Hierarchical Attention
1. Step Attention: Aggregates steps within a single walk.
2. Walk Attention: Aggregates multiple walks of the same type.
3. Scale Fusion: Aggregates short, long and TAWR representations.





Input: src_nodes, dst_nodes, timestamps
|
MultiScaleWalkSampler (returns walk indices/times/masks, no embeddings)
|
HCT (replace WalkEncoder)
 |-Looks up node features from SAM memory using walk indices
 |-Intra-walk encodings
 |-Co-Occurrence matrix from anonymized nodes
 |-Inter-walk transformer
 |-Returns walk-based node embeddings
|
Cobine base TGN embeddings
|
Link Predictor
