Component v1: dynamic Co-occurrence GNN
Component v2: v1 + causal temporal attention guided walk

H1 (Contextual Awareness): The proposed causal temporal-delta attention walk encoder (W-H1) captures more fine-grained temporal patterns than TAWRMAC's fixed-time encoding and neighbor co-occurrence embedding, resulting in superior representation quality with greater parameter efficiency.

H2 (Structural Dynamics): Attention‑guided walk generation (W‑H2) produces walks that are more informative and context‑aware than TAWRMAC's random TAWR walks, with a smaller computational footprint.

H3 (Memory Stability): The enhanced memory updating mechanism (W‑H3) alleviates embedding staleness more effectively than TAWRMAC's memory‑augmented GNN, particularly for long‑range interactions.

H4 (Generalization): The proposed method generalizes better than TAWRMAC across transductive and inductive settings under all three negative sampling strategies.

Compare your method against TAWRMAC and other state-of-the-art baselines on dynamic link prediction and node classification.

Evaluation Tasks
1. Dynamic Link Prediction:

Train/validation/test split: 60%/20%/20% chronological split (no information leakage)

Negative sampling during evaluation: All three TAWRMAC strategies: Random (-): negative edges sampled uniformly; Temporal (–): negative edges using same source node but different timestamps; Structural (–): negative edges using same structural context but different nodes, the most challenging setting that tests true structural understanding.

Metrics: Average Precision (AP) as primary; Area Under the ROC Curve (AUC‑ROC) and Hits@K metrics as secondary

Baselines
TAWRMAC – your primary target to beat

Strong temporal GNN baselines: TGN, TGAT, JODIE, DyRep

Recent transformer‑based methods: GraphMixer, DyGFormer, TCL

Abbreviations of your method: Prototype for validation without walk reduction, walk reduction only, full method

W‑H1 (contextual awareness encoder): replaces TAWRMAC's fixed‑time encoding + neighbor co‑occurrence with causal time‑delta attention walk encoder; retains TAWRMAC's original TAWR walk generation

W‑H2 (walk reduction + attention‑guided): replaces TAWRMAC's random TAWR walks with attention‑guided walks, but retains TAWRMAC's original time encoder

W‑H3 (full method): W‑H1 + W‑H2

Datasets
Select at least four real‑world dynamic graphs offering diverse structural and temporal characteristics:

Wikipedia – user‑article edit interactions, moderate size

Reddit – user‑post interaction sequences, high activity

LastFM – music listening records with temporal dynamics

MOOC – student‑course action sequences, educational interactions

All four are standard DGB/TGB datasets; other options include Social Evo., UCI, Enron, Flights, etc..

Evaluation Protocol
Run 10 independent trials with different random seeds

Report mean ± standard deviation across all runs

Apply statistical significance tests (paired t‑test or Wilcoxon signed‑rank) for each metric comparison

If TAWRMAC code not publicly released, implement it yourself as accurately as possible following the paper description

Experiment 2: Ablation Study
Objective
Determine which component drives performance improvement.

Configurations to Compare
Full method (W‑H1 + W‑H2)

No attention guidance (~W‑H2): retains W‑H1 attention encoder but samples walks as TAWRMAC (random walks with restart)

No time‑delta attention encoder (~W‑H1): retains walk reduction and attention‑guided walk generation but uses TAWRMAC's fixed‑time encoder

Expected outcomes
If W‑H1 is the primary driver: Full method ≈ No walk reduction > No attention guidance; if W‑H2 is the primary driver: Full method ≈ No attention guidance > No walk reduction; with likely complementary effects where both are necessary.

Experiment 3: Computational Efficiency Analysis
Objective
Quantify whether your method achieves superior performance with less computation—a key advantage.

Metrics to Report
Parameters: model parameter count (demonstrating your method is more compact)

Memory usage: peak GPU/CPU memory (for both training and inference)

Training time per epoch: seconds

Training throughput: processed links/edges per second (throughput = (batch size × number of batches)/total training time)

Inference latency: time for single batch of predictions

Walk generation time: time to generate all walks per batch (this should be strictly lower than TAWRMAC due to reduced 1 vs. 15 walks per node)

All metrics averaged across 10 runs with standard deviation reported. Plot performance vs. computation time (FLOPS or inference time) to visually demonstrate your efficiency advantage.

Experiment 4: Hyperparameter Sensitivity
Objective
Demonstrate robustness of your model.

Parameters to Vary
Walk lengths (short: 2/3/5; long: 6/10/15; TAWR: 5/8/12)

Number of walks (1–2 per scale – but your method uses 1; vary to see if >1 hurts/helps)

Time‑delta attention temperature (0.01, 0.05, 0.1, 0.5)

Memory dimension (64, 128, 256)

Learning rate (1e‑4, 5e‑4, 1e‑3)

Plot AP vs. each parameter (10 runs per configuration) to show smooth curves and comment on whether default parameters are near‑optimal. Also report convergence time across hyperparameter settings.

Experiment 5: Temporal Generalization Analysis
Objective
Prove your method better handles long‑range dependencies.

Protocol
Train each model on the first X% of the timeline and test on remaining timeline increments, e.g., 10%/10%/10% increments. For each, record AP and Macro F1. Plot performance vs. test time horizon—the model with slowest degradation has best long‑range understanding. Additionally, compute time‑distance correlation between training‑test gap and performance drop, and compute Spearman correlation between node activity and performance gap.

Experiment 6: Attention Interpretability Analysis
Objective
Qualitatively validate that attention‑guided walks follow meaningful temporal patterns.

Visualizations to Produce
Attention weight heatmaps over walk steps (rows: steps, columns: tokens) – demonstrate that attention tends to focus on recent interactions for rapidly changing nodes, but spreads over longer histories for stable nodes

Walk trajectory visualizations comparing TAWRMAC random walks vs. your attention‑guided walks (e.g., for a given source node, highlight selected paths)

Case studies of specific nodes: select low/medium/high activity nodes and show how historical attention weights change across steps

Comparison to TAWRMAC Claims
Your experiments must directly address TAWRMAC's claims:

TAWRMAC Claim	 Experimental Counter‑Evidence
Memory augmentation alleviates embedding staleness (Claim 1)	Show memory‑aware attention captures longer histories with fewer parameters (Experiment 5)
Neighbor co‑occurrence improves contextual awareness (Claim 2)	Show explicit attention to temporal history is more precise (Experiment 1)
TAWR mechanism captures structural dynamics (Claim 3)	Show attention‑guided walks are more informative (Experiment 2)
Outperforms SOTA on link prediction & node classification	Re‑run TAWRMAC under identical conditions to confirm; if public, use official code; if not, reimplement faithfully. Perform McNemar’s test for statistical significance on predictions
Reproducibility and Validation Standards
Open‑source all code: on GitHub with instructions, Dockerfile, and reproducible runs

Share hyperparameters: final hyperparameters + search grid for each dataset

Describe all hardware: GPU type, CPU, RAM, OS, and any optimizations used

Include randomness control: random seeds used, number of seeds, and variance estimates

Make negative sampling pipelines available: all three TAWRMAC strategies (random, temporal, structural) implemented consistently

Run baselines on identical hardware: no cherry‑picking baseline results from literature; re‑run all baselines in your environment

Use proper statistical testing: paired t‑test/Wilcoxon signed‑rank with significance threshold (α＝0.05) and effect sizes (Cohen’s d)

Following this protocol will allow to claim that your method not only outperforms TAWRMAC but does so for theoretically grounded reasons, with computational efficiency and interpretability advantages.

Here is our Active experiment repository : https://github.com/NilSagor/dyngr_rl

how should do experiment and analysis each component, ablation and param sweep