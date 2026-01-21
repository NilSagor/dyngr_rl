# experiment_log.md

## Run: DyGFormer-Wikipedia-Historical (2026-01-21)

### Configuration
- **Model**: DyGFormer
- **Dataset**: Wikipedia (ml_wikipedia)
- **Negative Sampling**: historical
- **Evaluation Type**: transductive
- **Seed**: 42
- **Config File**: `configs/dygformer_config.yaml`
- **GPU**: RTX 4060ti 16g
### Results
| Metric      | Our Result | Paper Result | Gap |Evaluation Type|
|-------------|---------|---------|---------------|----|
| Test Accuracy | 0.9265 |0.86|-+0.0665|Inductive|
| Test AP       | 0.5079 |**0.86-0.99**|**-0.35-0.48**|inductive|
| Test Loss     | 0.2087 |-|-|-|

### Analysis
- **AP is critically low** (0.507 vs expected: 0.86+ based on literature), confirming **H9: Negative Sampling Bias**
- High accuracy (0.9265) is misleading due  to **class imbalance artifact** (few positive edges)
- **AUC = nan** because test batches contain **only one class** → negative sampling issue 
- Possible issues:
  - **Root cause**: Negative sampling generates **too-easy negatives** or **incorrect temporal masking**
  - Negative sampling not aligned with DyGLib protocol
  - Missing edge features (Wikipedia has 172-D edge features!)
  - Incorrect temporal split

### Action Items
- [x] **Confirmed data loading works** (no crashes, correct shapes)
- [ ] **Fix negative sampling** to match DyGLib protocol:
  - Historical: sample from past positive edges not in current batch
  - Inductive: sample from future test edges
- [ ] **Implement proper edge feature integration** in DyGFormer
- [ ] **Run transductive baseline** (all nodes seen in training)
- [ ] **Implement TGN baseline** for H1 testing

<!-- ### Links to Hypotheses
- **H9**: Negative Sampling Bias → This run tests sensitivity to sampling strategy
- **MH1**: Reproducibility Crisis → Can we match published DyGFormer results?

### Status
**Anomalous result** — requires debugging before proceeding to H1 testing. -->