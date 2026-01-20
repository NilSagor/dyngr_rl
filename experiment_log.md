# experiment_log.md

## Run: DyGFormer-Wikipedia-Historical (2026-01-20)

### Configuration
- **Model**: DyGFormer
- **Dataset**: Wikipedia (ml_wikipedia)
- **Negative Sampling**: historical
- **Evaluation Type**: transductive
- **Seed**: 42
- **Config File**: `configs/dygformer_config.yaml`
- **GPU**: RTX 4060ti 16g
### Results
| Metric      | Our Result | Paper Result | Gap |
|-------------|---------|---------|---------------|
| Test Accuracy | 0.9978 |0.86|-0.3538|
| Test AP       | 0.5062 |-|-|
| Test Loss     | 0.0074 |-|-|

### Analysis
- **Unexpectedly low AP** (expected: >0.86 based on literature)
- High accuracy suggests **class imbalance artifact**
- Possible issues:
  - Negative sampling not aligned with DyGLib protocol
  - Missing edge features (Wikipedia has 172-D edge features!)
  - Incorrect temporal split

### Action Items
- [ ] Verify data loader uses **edge features** from `.npy`
- [ ] Implement **TGN baseline** for comparison
- [ ] Validate negative sampling strategy matches Poursafaei et al. (2022)
- [ ] Compare with **random** and **inductive** sampling

<!-- ### Links to Hypotheses
- **H9**: Negative Sampling Bias → This run tests sensitivity to sampling strategy
- **MH1**: Reproducibility Crisis → Can we match published DyGFormer results?

### Status
**Anomalous result** — requires debugging before proceeding to H1 testing. -->