How to combine them for statistical analysis (easy)
Since every CSV has the same column structure, you can aggregate them with a 3‑liner in Python (run it inside a notebook in experiments/notebooks/):

```python
    
    import pandas as pd
    from pathlib import Path

    # Find all individual result CSVs
    csv_files = list(Path("experiment_framework/outputs").rglob("all_results.csv"))
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df.to_csv("aggregated_results.csv", index=False)


Now we have a single DataFrame with columns: model, dataset, seed, test_ap, test_auc, training_time, etc.
we can then group, filter, and run statistical tests (via scipy.stats or statsmodels) directly on it.
