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



1. Explicit co-occurrence captures structural patterns that walks approximate indirectly

2. Causal time-delta attention directly models the spacing of interactions — something TAWRMAC only does via walk bias, which is weaker and less direct.

3. Hierarchical fusion lets the model decide per node-pair how much to trust structure (co-occurrence) vs. anonymous walk patterns vs. recency. TAWRMAC has no such explicit gating.

4. co-occurrence on dense graphs, recency on bursty graphs



Component Details
1.1 TAWR Walk Sampler (Single Type)
Only TAWR walks (no short/long) – restart provides natural multi-scale

walk_length = 8, num_walks = 8 per node

Restart probability: $\rho_u(\tau) = \sigma(\mathbf{w}_\rho^\top[\mathbf{m}u(\tau)|\Phi(\tau)] + b\rho)$

1.2 Light HCT Encoder
Intra-walk: 1 Transformer layer, 1 head, d_model=128

Inter-walk: 1 Transformer layer, 1 head

No co-occurrence bias inside HCT (moved to explicit Co-GNN)

Mean pooling for walk aggregation (no attention pooling)

1.3 Co-GNN (Explicit Co-occurrence)
Precompute first-hop co-occurrence matrix once per epoch:
Cuv=∣N(u)∩N(v)∣∣N(u)∣⋅∣N(v)∣Cuv =∣N(u)∣⋅∣N(v)∣∣N(u)∩N(v)∣
 

2-layer GNN:
cu(1)=ReLU(W1⋅[xu;∑v∈N(u)Cuvxv])
cu(1)=ReLU(W1 ⋅[xu ;∑v∈N(u) Cuv xv ])
cu=W2cu(1)cu =W2 cu(1)
​

Output dimension = hidden_dim (128)

1.4 Causal Temporal Attention (Replaces ST-ODE)
Maintain per-node history buffer (last 32 interactions)

Time-delta encoding:
δti=MLP([sin(ω1Δi),cos(ω1Δi),
…
]
)
,
Δ
i
=
t
curr
−
t
i
δ 
t 
i
​
 
​
 =MLP([sin(ω 
1
​
 Δ 
i
​
 ),cos(ω 
1
​
 Δ 
i
​
 ),…]),Δ 
i
​
 =t 
curr
​
 −t 
i
​
 

History embedding: $\mathbf{h}_i = \mathbf{m}u(t_i) + \boldsymbol{\delta}{t_i}$

Causal attention (only past positions):
α
i
j
=
exp
⁡
(
h
i
⊤
W
Q
W
K
⊤
h
j
/
d
)
∑
p
<
i
exp
⁡
(
h
i
⊤
W
Q
W
K
⊤
h
p
/
d
)
α 
ij
​
 = 
∑ 
p<i
​
 exp(h 
i
⊤
​
 W 
Q
​
 W 
K
⊤
​
 h 
p
​
 / 
d
​
 )
exp(h 
i
⊤
​
 W 
Q
​
 W 
K
⊤
​
 h 
j
​
 / 
d
​
 )
​
 
z
u
temp
=
∑
j
<
i
α
i
j
W
V
h
j
z 
u
temp
​
 =∑ 
j<i
​
 α 
ij
​
 W 
V
​
 h 
j
​
 

1.5 Hierarchical Cross-Attention Fusion
Query: walk embedding $\mathbf{w}_u$ (from HCT)

Key/Value: concatenation $[\mathbf{c}_u; \mathbf{z}_u^{\text{temp}}]$

Multi-head cross-attention (2 heads):
f
u
=
CrossAttn
(
Q
=
w
u
,
K
=
V
=
[
c
u
;
z
u
temp
]
)
f 
u
​
 =CrossAttn(Q=w 
u
​
 ,K=V=[c 
u
​
 ;z 
u
temp
​
 ])

Residual connection: $\mathbf{f}_u = \mathbf{f}_u + \mathbf{w}_u$

1.6 SAM Memory Update (Prototype-based)
Prototypes: 5 per node (ablatable)

Update rule (same as original SAM):
s
u
(
t
)
=
(
1
−
β
u
)
m
u
(
t
−
)
+
β
u
∑
k
α
u
k
p
u
k
s 
u
​
 (t)=(1−β 
u
​
 )m 
u
​
 (t 
−
 )+β 
u
​
 ∑ 
k
​
 α 
u
k
​
 p 
u
k
​
 

Memory GRU (for online update after each batch):
m u(t)=GRU ([mu(t−);mv(t−);euv;Φ(Δt)],mu(t−))mu (t)=GRU([mu​ (t− );mv (t− );euv;Φ(Δt)],mu (t− ))

1.7 Link Predictor
Simple MLP: MergeLayer(hidden_dim, hidden_dim, 1)

Temperature-scaled sigmoid for probability

2. Ablation Experiment Settings
2.1 Datasets
Dataset	Nodes	Edges	Time span	Type
Wikipedia	9,227	157,474	30 days	Bipartite, rich text
Reddit	10,984	672,447	30 days	Bipartite, post interactions
MOOC	7,144	411,749	30 days	Student-course actions
LastFM	1,980	1,293,103	28 days	User-song listening
2.2 Evaluation Protocol
Transductive: all nodes seen during training

Inductive: 10% unseen nodes in test

Negative sampling: random (for main comparison), historical, inductive

Metrics: AP (Average Precision), AUC-ROC

Train/val/test split: 70/15/15 (chronological)

2.3 Ablation Variants
Variant	Components	Purpose
A0: TAWRMAC	Original TAWRMAC (baseline)	Reproduce SOTA
A1: + Co-GNN	TAWRMAC + explicit co-occurrence	Measure co-occurrence gain
A2: + Causal Attn	A1 + replace walk bias with time-delta attention	Measure temporal modeling gain
A3: + SAM	A2 + prototype memory (instead of raw GRU)	Measure memory stability gain
A4: HiCoST-R (full)	All components + hierarchical fusion	Complete model
A5: No Co-GNN	Full model without Co-GNN	Ablate co-occurrence
A6: No Temporal	Full model without causal attention	Ablate temporal signal
A7: No HCT	Full model with mean pooling instead of HCT	Ablate hierarchical walks
A8: ST-ODE (old)	Replace causal attention with ST-ODE	Compare to old design
2.4 Hyperparameters (Fixed Across All Ablations)
yaml
model:
  hidden_dim: 128
  memory_dim: 128
  time_dim: 64
  dropout: 0.2
  num_prototypes: 5
  walk_length: 8
  num_walks: 8
  walk_temperature: 0.1
  history_len: 32
  cooc_normalize: True

training:
  batch_size: 256
  learning_rate: 1e-4
  weight_decay: 1e-5
  max_epochs: 50
  warmup_epochs: 5
  label_smoothing: 0.1
  neg_sample_ratio: 5
  hard_neg_threshold: 0.7
2.5 Expected Results (Wikipedia Transductive Random)
Variant	AP	AUC	Training time/epoch	Δ vs TAWRMAC
A0: TAWRMAC	0.840	0.796	~2 min	–
A1: + Co-GNN	0.852	0.808	~2.5 min	+0.012 AP
A2: + Causal Attn	0.861	0.817	~3 min	+0.021 AP
A3: + SAM	0.873	0.828	~3.5 min	+0.033 AP
A4: HiCoST-R (full)	0.886	0.841	~4 min	+0.046 AP
A5: No Co-GNN	0.874	0.830	~3.5 min	–0.012 from full
A6: No Temporal	0.868	0.823	~3 min	–0.018 from full
A7: No HCT	0.852	0.809	~2.5 min	–0.034 from full
A8: ST-ODE (old)	0.802	0.761	~15 min	–0.038 from full
2.6 Expected Results (Reddit Transductive Random)
Variant	AP	AUC	Training time/epoch
TAWRMAC	0.985	0.992	~3 min
HiCoST-R (full)	0.991	0.996	~5 min
3. Implementation Checklist for HiCoST-R
New Files to Create
component/co_gnn.py – explicit co-occurrence GNN

component/causal_temporal_attention.py – time-delta + causal attention

component/hierarchical_fusion.py – cross-attention fusion module

Modifications to Existing
multi_swalkv2.py – keep only TAWR walks (remove short/long)

hct_modulev2.py – remove co-occurrence bias, reduce to 1 layer/1 head

sam_modulev2.py – keep as is (already good)

hicost_v4.py – integrate new components, remove ST-ODE

Training Script Snippet
python
# Run all ablations
for variant in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']:
    config = load_config(f'configs/ablation_{variant}.yaml')
    model = HiCoSTR(config)
    trainer = pl.Trainer(max_epochs=50, callbacks=[ModelCheckpoint(monitor='val_ap')])
    trainer.fit(model, train_dataloader, val_dataloader)
    results[variant] = trainer.callback_metrics
4. Scientific Claims for Paper
If the expected results hold, you can claim:

Explicit co-occurrence improves AP by +1.2% over TAWRMAC (A1 vs A0)

Causal temporal attention adds another +0.9% (A2 vs A1)

SAM prototypes add +1.2% (A3 vs A2)

Hierarchical fusion adds final +1.3% (A4 vs A3)

Total improvement: +4.6% AP over TAWRMAC on Wikipedia

ST-ODE is harmful – removing it and replacing with lightweight attention recovers 8% AP