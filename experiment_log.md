# experiment_log.md

## Run: DyGFormer-Wikipedia-Historical (2026-01-21)

### Configuration
- **Model**: DyGFormer
- **Dataset**: Wikipedia (ml_wikipedia)
- **Negative Sampling**: random/historical/inductive
- **Evaluation Type**: transductive/inductive
- **Seed**: 42
- **Config File**: `configs/dygformer_config.yaml`
- **GPU**: RTX 4060ti 16g
### Results
|Model Name|Dataset  |EvaluationType|Sampling Strategy|Test AP|TEST AUC|Test Acc|Test Loss|Notes  |
|----------|-------  |----------    |---------        |-------|----    |--------|---------|---    |
|          |         |              |Random           |0.9811 |0.9711  |0.9239  |0.2147   |       |
|          |         |              |Historical       |0.4794 |0.4376  |0.4276  |5.0051   |       |   
| DyGFormer|Wikipedia|Inductive     |inductive        |0.9801 |0.9706  |0.9260  |0.2285   |       | 
|          |         |              |Random           |0.9811 |0.9711  |0.9239  |0.2147   |       |
|          |         |              |Historical       |0.4794 |0.4376  |0.4276  |5.0051   |       |
|          |         |Transductive  |inductive        |0.9801 |0.9706  |0.9260  |0.2285   |       |


### Analysis
- Extreme Performance Disparity by Sampling Strategy
  Random NSS: AP ≈ 0.98 (near-perfect)
  Historical NSS: AP ≈ 0.48 (worse than random)
  50-point AP gap confirms that negative sampling strategy dramatically impacts model rankings
- Evaluation Type Independence
  Both transductive and inductive settings show identical patterns
- Historical NSS Failure Mode
  AP = 0.48 < 0.5 means the model performs worse than random guessing
  High loss (5.0) indicates severe optimization instability

### Action Items
- [x] **Confirmed data loading works** (no crashes, correct shapes)
- [ ] **Fix negative sampling** to match DyGLib protocol:
  - Historical: sample from past positive edges not in current batch
- [ ] **TGN Baseline Implementation**
- [ ] **TAWRMAC 2025 Baseline Implementation**
- [ ] **Implement TGN baseline** for H1 testing

<!-- ### Links to Hypotheses
- **H9**: Negative Sampling Bias → This run tests sensitivity to sampling strategy
- **MH1**: Reproducibility Crisis → Can we match published DyGFormer results?

### Status
**Anomalous result** — requires debugging before proceeding to H1 testing. -->