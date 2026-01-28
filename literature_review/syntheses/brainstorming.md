Chapter I: The "Temporal Truth" Thrust
Experiments: Look-Ahead Filtering + Ranking-Aware Curriculum.

Story: "Current evaluation is broken because we treat future positives as negatives (Positive Shift). We propose a filtering mechanism that respects the temporal horizon and a loss function that optimizes for ranking rather than binary classification."

Datasets: Wikipedia, Reddit, LastFM (High-frequency interaction datasets where "Positive Shift" is most visible).



Chapter II: The "Hardness & Representation" Thrust
Experiments: Dual-Track Curriculum + Adversarial Disentanglement.

Story: "Negatives aren't just 'not edges'; they are either 'structurally far' or 'temporally mismatched.' We use disentangled adversarial mining to force the model to learn both dimensions separately, solving the representation bottleneck in sparse graphs."

Datasets: UCI, Enron, MOOC (Lower density datasets where "Positive Sparsity" makes structure learning difficult).


Chapter III: The "Scalability & Robustness" ThrustExperiments: Vector-DB Acceleration + Adaptive Annealing.Story: "Sophisticated negative mining is computationally expensive. We introduce a Vector-DB indexed sampling strategy that maintains $O(\log N)$ complexity even with annealing hardness, making advanced TGNNs usable on real-world web-scale graphs."Datasets: Flights, Large-scale benchmarks (1M+ nodes)