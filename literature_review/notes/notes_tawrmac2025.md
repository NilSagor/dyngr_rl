# {TAWRMAC: A Novel Dynamic Graph Representation Learning Method} - DBLP:journals/corr/abs-2510-09884


## 1. MetaData
- **Citation Key**: DBLP:journals/corr/abs-2510-09884
- **Venue**: arXiv (preprint, submitted to ICLR/NeurIPS-level venue)
- **Code**: https://anonymous.4open.science/r/tawrmac-A253/README.md

## Problem & Motivation


## Method Mechanism
- Key Diagram:


- **Core Equation**: 
    -  Final node embedding:
        $$
        h_{u}(t) = h_{u}^{(L-1)}(t)
        $$

    - Neighbor Co-occurrence Embedding (NCE)
        $$ 
        ce_{u}(t) = \text{MLP}(nc_{u})[:,0] + \text{MLP}(nc_{u}(t)[:,1])
        $$

    - Learnable restart probability (Eq.3):
        $$ 
        \text{pr}_{u}(t) = \text{MLP}(h_{u}(t))
        $$ 
    
    - Final Embedding:
        $$ \text{emb}_{u}(t) = [h_{u}(t)||ce_{u}(t)||enc(WR_{u})||[pr_{u}(t)] $$
    - Memory Update:
    $$
    m_{u} = \text{RNN}([m_{u}(t^{-})||m_{v}(t^{-})|| \varphi_{1} (\Delta t) || e_{uv}(t)], m_{u}(t^{-}))
    $$
    
    

- **Trick**:
    - 


#### Technical Approach
```
Input: Temporal Graph
|
\/
MAE Module: [Node Memory x Time Encoding x Node Features]
|
\/
NCE Module: [Neighbor Co-occurrence Matrix ->MLP processing]
|
\/
Final Embedding: [MAE Output x NCE Output]
|
\/
Downstream Task: (Link Prediction/Node Classification)
```


## Results & Critical Gaps 




## Connections
- Cites
- Contradiction
    - Challenges DyGFormer's assumption that pure attention + first-hop is sufficient $\rightarrow$ adds memory + walk-based structure.
    - Contradicts FreeDyG by showing frequency-domain isn't necessary for SOTA -- structural + memory cues suffice. 



## 6. Atcion Items
- Re-implements: High Priority - code available; align with Tier 1 in core8_list.md
- Baseline candidate: Essential baseline for H1(Memory-Attention Complementarity) and G7 (Multi-hop Gap)
- Idea Spark:
    - Can we replace TAWR's restart mechanism with multi-hop GNN sampler?
    - Could NCE be exteneded to 2-hop co-occurrence without explosion?
    - Does TAWRMAC's pr(t) correlate with optimial neighborhood size? (Link to G8:Adaptive Architecture)

## 7. Key Insights
- Surprising Finding: Fixed time $\varphi_{2}$ significantly improves stability -- contrary to trend of fully learnable encodings.
- Key Insight: The synergy of three orthogonal signals (memory, co-occurrence, walks )matters more than any single component.
- Modular design enables both strong performance and interpretability-- ideal for hypothesis-driven research.