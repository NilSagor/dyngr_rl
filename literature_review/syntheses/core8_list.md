# Core 8 Dynamic Graph Models (2023-2025)

### 1. TAWRMAC 2025 #DBLP:journals/corr/abs-2510-09884

**Title**: A Novel Dynamic Graph Representation Learning Method
- **Key Innovation**: Memory-augmented embedding with neighbor co-occurrence encoding
- **Architecture**: MAE (Memory-Augmented Embedding) + NCE (Neighbor Co-occurrence Embedding)
- **Strengths**:
    - Faster Convergence than DyGFormer
    - Better computational efficiency
    - Captures structural patterns effectively
- **Limitations**: Limited to first-hop neighbors
- **Datasets**: Wikipedia, MOOC, UCI
- **Performance**: State-of-the-art on multiple benchmarks



## Implementation Priority

### Tier 1 (essential)
1. **TGN** - Foundational model, must-implement baseline
2. **DyGFormer** - Current state-of-the-art transformer approach
3. **TAWRMAC** - Latest and most efficient

### Tier 2 (Important)



## Key Design Patterns Identified

### 1. Memory-Augmented Architectures
- **Models**: TGN, TAWRMAC, EdgeBank
- **Pattern**: Maintain compressed node history representations
- **Advantages**: Efficient temporal pattern capture

### 2. Attention Mechansims

- **Models**: DyGFormer, TAWRMAC (co-occurrence)
- **Patterns**: Use attention for temporal/structural dependencies
- **Advantages**: Flexible relationship modeling

### 3. Time Encoding
- **Models**: All except EdgeBank
- **Patterns**: Map Timestamps to vector representations
- **Methods**: Sinusoidal, learnable, frequency-based

### 4. Neighborhood Sampling
- **Models**: TGN, DyGFormer, CTGN
- **Patterns**: Sample temporal neighborhoods for aggregation
- **Challenge**: Balancing coverage and efficiency

