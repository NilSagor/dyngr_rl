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




Multi-Scale Walk Sampler

code: https://github.com/NilSagor/dyngr_rl/blob/main/experiment_framework/src/models/enhanced_tgn/component/multi_swalkv2.py

The primary goal of the Multi-Scale Walk Sampler is to generate a set of temporal walks for a node $u$ at time $t$ that collectively capture:
- Local, fine-grained structure (e.g., immediate neighbors, recent interactions).
- Global, coarse-grained structure (e.g., community patterns, long-range dependencies).
- Exploratory behavior (e.g., nodes that frequently form new connections beyond their immediate neighborhood).

It achieves this by sampling three distinct types of walks: Short Walks, Long Walks, and TAWR-style Walks with Restart.

Step 1: Temporal Neighbor Sampling with Exponential Bias

Before defining the walk types, we need a core sampling function. For any node $x$ at a current walk time $\tau$, we sample a neighbor $y$ from its valid historical neighbors $\mathcal{N}_{\tau}(x)$ with a probability that favors more recent interactions. 

$$
P(y, \tau' \mid x, \tau) = \sum_{z \in N_{\tau}(x)} \frac{\exp \left( \beta \frac{\tau' - \tau_{max}}{\cdot} \right)}{\exp \left( \beta \frac{\tau_z - \tau_{max}}{\cdot} \right)}
$$

Where:
-$\tau'$ is the timestamp of edge $(x, y)$.
-$\tau_{\text{max}} = \max{\tau_z : z \in \mathcal{N}_{\tau}(x)}$ (the most recent neighbor timestamp).
-$\beta$ is a temperature parameter controlling the bias strength (small $\beta$ strongly favors the most recent neighbor).

This ensures walks are temporally coherent and emphasize recent, likely more relevant, interactions.

Step 2: Short Walks (Capturing Local Structure)
Short walks are designed to capture the immediate, fine-grained neighborhood of node $u$. They are shallow and stay close to $u$.

For $r = 1$ to $R_s$:

-Initialize walk $W = [(u, t)]$.
-Set current node $x = u$, current time $\tau = t$.
-For step $i = 1$ to $L_s$:
	a. If $\mathcal{N}_{\tau}(x) = \emptyset$, break.
	b. Sample next node $y$ using the temporal bias $P(y, \tau' | x, \tau)$ from Step 1.
	c. Append $(y, \tau')$ to $W$.
	d. Update $x = y$, $\tau = \tau'$.
-Add $W$ to $\mathcal{W}_u^{\text{short}}$.

Short walks approximate a truncated, temporally-biased random walk on the graph. They explore the $L_s$-hop neighborhood but are likely to stay in high-density regions due to the temporal bias. They provide the "local texture" for HCT.



Step 3: Long Walks (Capturing Global Structure)
Long walks are designed to explore further from $u$, capturing broader community structures and longer-range dependencies. They are deeper and can venture into other parts of the graph.

For $r = 1$ to $R_l$:

- Initialize walk $W = [(u, t)]$.
- Set current node $x = u$, current time $\tau = t$.
- For step $i = 1$ to $L_l$:
a. If $\mathcal{N}_{\tau}(x) = \emptyset$, break.
b. Sample next node $y$ using the temporal bias $P(y, \tau' | x, \tau)$ from Step 1.
c. Append $(y, \tau')$ to $W$.
d. Update $x = y$, $\tau = \tau'$.
- Add $W$ to $\mathcal{W}_u^{\text{long}}$.

The only difference from short walks is the length $L_l > L_s$. This allows the walk to take more steps and potentially reach nodes far from $u$, capturing global patterns. However, the temporal bias still applies at each step, so the walk remains temporally coherent.


Step 4: TAWR-style Walks with Restart (Capturing Exploratory Behavior). 
TAWR-style walks introduce a learnable, time-dependent, node-specific restart probability $\rho_u(\tau)$. At each step, instead of always continuing the walk, the walk may "restart" from the original source node $u$, allowing it to explore multiple "branches" from the source.

First, we define the restart probability for node $u$ at time $\tau$ (the current walk time):

$$
\rho_u(\tau) = \sigma \left( w_{\rho}^T \cdot [m_u(\tau) \parallel \Phi(\tau)] + b_{\rho} \right)
$$

Where:

- $\sigma$ is the sigmoid function, ensuring $\rho_u(\tau) \in (0, 1)$.

- $\mathbf{m}_u(\tau)$ is the memory state of node $u$ at time $\tau$ (from the SAM component).

- $\Phi(\tau)$ is the fixed time encoding.

- $\mathbf{w}\rho$ and $b\rho$ are learnable parameters.

This probability is node-specific (different nodes have different restart tendencies), time-dependent (a node may become more exploratory at certain times), and learnable (the model can discover optimal restart strategies from data).



TAWR Walk Generation:

For $r = 1$ to $R_r$:
1. Initialize walk $W = [(u, t)]$.
2. Set current node $x = u$, current time $\tau = t$.
3. For step $i = 1$ to $L_r$:
a.If $\mathcal{N}_{\tau}(x) = \emptyset$, break.
b. With probability $\rho_u(\tau)$: Restart. Set $x = u$ (reset to source node), $\tau = t$ (reset to original time). Append a special restart token $(u, t, \text{restart=true})$ to the walk to mark this event.
c. With probability $1 - \rho_u(\tau)$: Continue normally.
i. Sample next node $y$ using the temporal bias $P(y, \tau' | x, \tau)$ from Step 1.
ii. Append $(y, \tau')$ to $W$.
iii. Update $x = y$, $\tau = \tau'$.
4. Add $W$ to $\mathcal{W}_u^{\text{tawr}}$.

Step 5: Walk Anonymization (Crucial for Inductive Learning)

Before passing the walks to HCT, we must anonymize them.
For each walk set $\mathcal{W}_u$, we:

Identify all unique nodes $\mathcal{V}_{\text{walk}}$ appearing in the walks.

For each walk $W \in \mathcal{W}u$, replace each node identity with an anonymized identifier based on its first occurrence time and frequency across all walks from $u$. This creates anonymized walks $W{\text{anon}}$.

For TAWR walks, the restart token is treated as a special anonymized node with its own identifier.
This anonymization ensures that the representations learned by HCT are isomorphism-aware and inductive—they depend on structural roles, not specific node IDs.


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

code: https://github.com/NilSagor/dyngr_rl/blob/main/experiment_framework/src/models/enhanced_tgn/component/hct_modulev2.py

Context: HCT's hierarchical approach and multi-scale walks capture neighbor correlations from local to global scales, creating rich, context-aware embeddings.

Addresses limited contextual awareness.

This component extends DyGFormer's neighbor co-occurrence idea and TAWRMAC's NCE to capture higher-order correlations. HCT operates on the sets of walks sampled by the multi-scale sampler. It first uses a Transformer to learn embeddings for nodes within a single walk (intra-walk dependencies). Then, another Transformer aggregates information across all walks for a given node (inter-walk dependencies). A novel co-occurrence matrix is computed not just for first-hop neighbors, but for the anonymized node roles appearing in the walks, capturing deeper structural patterns (e.g., "node that appears as a common second neighbor").


Step 1: Input Representation for a Single Walk

For each step $i$ in walk $W_u^{(r)}$, we first create an input embedding $\mathbf{z}_i^{(r)}$ that combines the anonymized node feature and the time encoding.

$$z_i^{(r)} = A(a_i^{(r)}) + \Phi(t_i^{(r)})$$

This gives us a sequence $\mathbf{Z}^{(r)} = [\mathbf{z}_1^{(r)}, \mathbf{z}2^{(r)}, ..., \mathbf{z}L^{(r)}] \in \mathbb{R}^{L \times d{\text{model}}}$ for walk $r$, where $d{\text{model}}$ is the Transformer's hidden dimension.

Step 2: Intra-Walk Transformer (Capturing Local Context)

We apply a Transformer encoder to each walk independently. This allows nodes within a walk to attend to each other, capturing the sequential and structural dependencies along the walk.

For walk $r$, we compute:

$$ H^{(r)} = \text{TransformerEncoder}_1(Z^{(r)}) $$

Where $\mathbf{H}^{(r)} \in \mathbb{R}^{L \times d_{\text{model}}}$ is the sequence of context-aware embeddings for each step in walk $r$. The TransformerEncoder uses multi-head self-attention, where for a single head, the attention from position $i$ to position $j$ is:

$$ \alpha_{ij}^{(r)} = \text{softmax} \left( \frac{(z_i^{(r)} W_Q^1)(z_j^{(r)} W_K^1)^T}{\sqrt{d_k}} \right) $$

This allows the model to learn, for example, that the second node in a walk is often influenced by the first node, regardless of the specific node IDs.

After the intra-walk Transformer, we aggregate each walk into a single vector, e.g., by mean-pooling:
	
$$w_u^{(r)} = \text{MeanPool}(H^{(r)}) \in \mathbb{R}^{d_{model}}$$

Now, we have a set of walk embeddings for node $u$: $\mathcal{E}_{u} = {\mathbf{w}_u^{(1)}, \mathbf{w}_u^{(2)}, ..., \mathbf{w}_u^{(R)}}$.



Step 3: Co-occurrence Matrix Construction

We construct a co-occurrence matrix $\mathbf{C}_u \in \mathbb{R}^{R \times R}$ that captures how often the same anonymized roles appear in similar positions across different walks. This goes beyond simple neighbor co-occurrence (as in DyGFormer) to higher-order structural patterns.

For two walks $r$ and $s$, we compute their co-occurrence score as:

$$ C_u[r, s] = \sum_{i=1}^{L} \sum_{j=1}^{L} \mathbb{I}(a_i^{(r)} = a_j^{(s)}) \cdot \kappa(i, j) $$

Where:$\mathbb{I}(a_i^{(r)} = a_j^{(s)})$ is an indicator function that is 1 if the anonymized node at position $i$ in walk $r$ is the same as the anonymized node at position $j$ in walk $s$. $\kappa(i, j)$ is a positional kernel that weighs co-occurrences based on how similar the positions are. For example:

$$ \kappa(i, j) = \exp\left(-\frac{|i - j|^2}{\sigma^2}\right) $$

This gives higher weight when the same anonymized node appears at similar positions in both walks (e.g., both as the starting node, both as the second node).

This matrix $\mathbf{C}_u$ captures rich structural information: if two walks share many of the same anonymized nodes in similar positions, they likely represent similar structural patterns.


Step 4: Inter-Walk Transformer with Co-occurrence Bias (Capturing Global Context)
we apply a second Transformer to model interactions between the $R$ walks of node $u$. This Transformer uses the co-occurrence matrix $\mathbf{C}_u$ as a bias in its attention mechanism, encouraging walks with similar structural patterns to attend more to each other. First, we compute the base attention between walks:

$$ \text{AttentionBase}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} \right) V $$

Where $\mathbf{Q} = \mathcal{E}_u \mathbf{W}_Q^2$, $\mathbf{K} = \mathcal{E}_u \mathbf{W}_K^2$, and $\mathbf{V} = \mathcal{E}_u \mathbf{W}_V^2$. We then incorporate the co-occurrence bias by adding it to the attention logits before the softmax:

$$ A_u = \text{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + \gamma \cdot C_u \right) $$

Where $\gamma$ is a learnable scalar that controls the strength of the co-occurrence bias. The output of this inter-walk Transformer is:

$$ E'_u = A_u V \in \mathbb{R}^{R \times d_{model}} $$

This is a refined set of walk embeddings, where each walk's representation is now informed by other walks that share similar structural patterns.

Step 5: Pooling to Final Node Embedding
Finally, we pool the refined walk embeddings to produce a single embedding for node $u$:

$$ h_u = \text{Pooling}(E'_u) \in \mathbb{R}^{d_{model}} $$

The pooling can be a simple mean, or a more sophisticated attention-based pooling that learns to weight walks based on their importance.





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



Time Encoder
Mulit-Scale Walk Sampler
Stability-Augmented Memory
Hierarchical Co-occurrence Transformer
Spectral-Temporal ODE
Mutual Refinement
Hard Negative Mining



Experiment:

Datasets:
Baseline:





