1. Time Encoder - Fixed trigonometric time encoding 
2. MultiScale Walk Sampler - generate short, long and TAWR
3. SAM- Prototype-base memory with gated updates
4. HCT- Intra walk and Inter walk encoding with co-occurrence
5. STODE - Continuous time evolution with spectral regularization
6. Mutual Refinement - Bidirectional cross-attention refinement
7. Pooling - AttentionPooling Attention based pooling over walks


Input: src_nodes, dst_nodes, timestamps

&#8595;
------------------------------------------------------
Multi-Scale Walk module
-Walk sampler ->walk for src/dst (short, long, tawr)
------------------------------------------------------

&#8595;
------------------------------------------------------
Walk Feature Fusion
combine
-Temporal walk pattern
-Structural walk pattern
-Edge time features
output: fused walk [B, num_walks, walk_len, fused_dim]
------------------------------------------------------

&#8595;
------------------------------------------------------


&#8595;
------------------------------------------------------
Mutual RefineMent & Pooling
[Mutual Refinement]
-Bidirectional cross-attention: src_walks <-> dst_walks
-Adaptive gating for information exchange

[Hierarchical Pooling]
-Pool within walk types (attention/mean/max)
-Fuse across types (short/long/tawr)
-output: refined_src_emb, refined_dst_emb [B, d_model]
-----------------------------------------------------------------

&#8595;
-------------------------------------------------------
Link Predictor Module
[Link Predictor] - MLP ([refined_src_emb, refined_dst_emb])-> Link Probability
------------------------------------------------------
&#8595;

output: Link Prediction


Stability-Augmented Memory (SAM)
Addresses embedding staleness and instability.

Instead of a single memory vector, SAM maintains a small set of prototype states for each node, learned via a temporal variational autoencoder (VAE). When a new interaction occurs, the node's memory is updated by attending to these prototypes based on the current context. This prevents the memory from drifting too far and becoming stale. It uses the fixed time encoding praised in the paper for stability, but combines it with a gating mechanism that learns to blend the old memory state with a candidate state derived from the interaction and prototypes.


Temporal graph networks must maintain per‑node historical states that evolve with interactions. However, a single memory vector updated at every event (as in TGN) is prone to noise and drift, leading to unstable embeddings. The Stability‑Augmented Memory (SAM) module addresses this by representing each node’s state not as a single vector but as a small set of learnable prototype vectors. These prototypes capture stable, long‑term modes of the node’s behaviour, while an attention‑based mechanism selects the relevant prototypes for the current context. A learned gate then blends the old memory with the prototype‑derived candidate, producing a new memory that is both stable and expressive.

Query Formation

At the time of an interaction, we first form a query that encodes the current context by combining the node’s current memory, the edge features, and the time of the event:


Candiate Memory
The candidate memory is a weighted combination of the prototypes, using the attention weights:

$$ \tilde{s}_u(t) = \sum_{k=1}^{K} \alpha_u^k(t) p_u^k $$

This candidate represents the “ideal” state suggested by the current context, grounded in the node’s learned stable patterns.


Gate Update with Temporal Context

A final update gate determines how much of the old memory should be retained versus replaced by the candidate. The gate depends on the old memory, the candidate, and the time encoding:
$$
\beta_u(t) = \sigma \left( W_{\beta} \cdot [m_u(t-1) \parallel \tilde{s}_u(t) \parallel \Phi(t)] + b_{\beta} \right)
$$

where $\sigma$ is the sigmoid function $W_{\beta} \in \mathbb{R}^{1 \times (2d + d_{\tau})}$ and $b_{\beta} \in mathbb{R}$ are learnable. The gate value lies in $[0,1]$. The new stabilised memory is then computed as a convex combination: 

$$s_u(t) = (1 - \beta_u(t)) \odot m_u(t-1) + \beta_u(t) \odot \tilde{s}_u(t)$$

This $s_{u}(t)$ becomes the raw memory for future interactions (i.e. $m_{u}(t^{-})$ for the next event).


Time Encoding
we employ a fixed (non‑learnable) time encoding $\phi (t)$
based on sinusoidal functions. This provides a stable representation of time that does not drift during training, further contributing to the overall stability of the memory updates.

Training 
During training, the prototypes are treated as ordinary parameters and receive gradients from the loss function through the attention mechanism and the final node embeddings. Because prototypes are shared across many interactions of the same node, they converge to represent the node’s persistent roles or states. The memory update itself is part of the forward pass; gradients flow through the update equations, allowing the gating mechanism to be learned as well.

During inference, we may wish to obtain a stabilised memory for a node without actually updating the stored raw memory (e.g., when the node is not involved in an interaction). This can be done by running the same SAM cell with the current raw memory, edge features (if any), and time encoding, but discarding the output $s_{u}(t)$ or using it as a transient embedding. In our implementation, the method get_stabilized_memory performs this forward pass without modifying the stored memory.


SAM provides enriched node embeddings that are fed to the subsequent components of the framework: the Multi‑Scale Walk Sampler gathers temporal neighbourhoods, the Hierarchical Co‑occurrence Transformer (HCT) captures multi‑scale dependencies, and the ST‑ODE models continuous‑time dynamics. By supplying stable yet adaptive node states, SAM ensures that the downstream modules work on representations that are robust to noise and reflect meaningful long‑term patterns.




Hierarchical Co-occurrence Transformer (HCT)
Context: HCT's hierarchical approach and multi-scale walks capture neighbor correlations from local to global scales, creating rich, context-aware embeddings.

Addresses limited contextual awareness.

This component extends DyGFormer's neighbor co-occurrence idea and TAWRMAC's NCE to capture higher-order correlations. HCT operates on the sets of walks sampled by the multi-scale sampler. It first uses a Transformer to learn embeddings for nodes within a single walk (intra-walk dependencies). Then, another Transformer aggregates information across all walks for a given node (inter-walk dependencies). A novel co-occurrence matrix is computed not just for first-hop neighbors, but for the anonymized node roles appearing in the walks, capturing deeper structural patterns (e.g., "node that appears as a common second neighbor").



Module: ST-ODE Module
code: https://github.com/NilSagor/dyngr_rl/blob/main/experiment_framework/src/models/enhanced_tgn/component/stode_module.py

Structural Dynamics:
Addresses capturing structural dynamics, especially without rich attributes. ST-ODE's spectral regularization explicitly models how representations should evolve through the graph structure, capturing dynamics even with weak attributes, as CAWN and TAWRMAC attempt but extend by adding a theoretically grounded evolution constraint.


Spectral-Temporal ODE Dual Path Dynamics

The SpectralTemporalODE module processes sequences of node‑wise observations (e.g., from random walks) that arrive at irregular timestamps. It maintains a hidden state per node and evolves it continuously between observations using an ODE. The evolution is regularized by a spectral term that encourages smoothness of the node representations over the graph structure (the SAM memory formation). The module can be stacked and integrated with other components.

GRUODECell: The GRU‑ODE variant (dh/dt = (1‑z)·(h̃‑h)) is chosen because it has been shown to work well for temporal graph data.

SpectralRegularizer: Computes the projection onto the leading k eigenvectors of the normalised Laplacian.

STODEFunc: Combines the base dynamics and the spectral regulariser.

STODELayer: Integrates the state from t_current to t_next using the ODE (with spectral regularization).

SpectralTemporalODE: Prepares observations from walk_encodings, walk_times, walk_masks in a vectorized way grouping by unique timestamps and averaging.


 


Let $\mathcal{H} \in \mathcal{R}^{N \times d}$ denote the hidden states of all $N$ nodes at continuous time $t$.  We model the evolution of $\mathcal{H}(t)$ between observations by an ordinary differential equation (ODE) that combines a learnable temporal dynamics with a graph‑based spectral regulariser.

Temporal Dynamics
we define: $f_{ode}: \mathcal{R} \times \mathcal{R} \rightarrow \mathcal{R}^{N \times d}$ as a neural network that captures the data-driven evolution. In practice, we implement $f_{ode}$ using a GRU‑inspired cell (or a multi‑layer perceptron) that takes the current state and time and outputs a derivative. This component is lightweight and fully differentiable.

Spectral regularizer.
To encourage smoothness of the node representations over the graph, we introduce a regularization term based on the graph Laplacian. Let $A$ be the adjacency matrix and $D$ the degree matrix. The normalized Laplacian is $ L = I - D^{- \frac{1}{2}}$. we compute its $k$ smallest eigenvalues and corresponding eigenvectors; the eigenvectors are stacked in $U \in \mathcal{R}^{N \times k}$. The regularizer is defined as:

$$ \mathcal{R}(H,A) = - \mu(I - UU^{T})H $$

where $\mu$  is a scalar (possibly learnable). The term $(I - UU^{T})$ projects $H$ onto the subspace orthogonal to the dominant eigenvectors, thereby damping high‑frequency components and promoting smoothness.

combined dynamics of full ODE is:

$$
\frac{dH}{dt} = f_{ode}(t, H) + \mathcal{R}(H,A).
$$

Integration and update

Given observations at times $t_{1}, t_{2}, \cdots $ with corresponding node features $H_{obs}^{(i)}$, we evolve the state from $t_{i}$ to $t_{i+1}$ by solving the ODE with a numerical integrator (e.g., Dormand‑Prince). Let $H_{ode}$ denote the state at $t_{i+1}$ obtained by integration. We then combine it with the observation using a gated mechanism:

$$
H(t_{i+1}) = \text{GRUCell}(H_{obs}, H_{ode})
$$

Because computing the full eigen‑decomposition at every step is prohibitive, we cache the eigenvectors and reuse them as long as the graph structure remains unchanged. A lightweight hash of the adjacency matrix detects changes, triggering a recomputation only when necessary. This caching makes the heavy spectral term practical in training and inference.



