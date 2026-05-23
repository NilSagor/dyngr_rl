# src/experiments/analysis/statistical_analyze.py
import pandas as pd
import numpy as np
from scipy import stats

def analyze_sensitivity_results(csv_path: str, metric: str = 'test_ap'):
    df = pd.read_csv(csv_path)
    
    # Filtering valid runs
    df = df[df[metric].notna() & (df.get('error', '').isna())]
    
    # Group by config name
    results = []
    for config_name, group in df.groupby('param_name'):        
        if group['param_value'].iloc[0].startswith('{'):
            subgroups = group.groupby('param_value')
        else:
            subgroups = [(group['param_value'].iloc[0], group)]
        
        for config_val, subgroup in subgroups:
            values = subgroup[metric].values
            if len(values) >= 2:  # Need >=2 for stats
                results.append({
                    'config': config_val if isinstance(config_val, str) else config_name,
                    'n_runs': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'ci_95': stats.t.interval(0.95, len(values)-1, 
                                            loc=values.mean(), 
                                            scale=stats.sem(values))[1] - values.mean()
                })
    
    summary_df = pd.DataFrame(results).sort_values('mean', ascending=False)
    
    # Statistical significance test (paired t-test if same seeds)
    if len(summary_df) >= 2:
        best = summary_df.iloc[0]
        for idx, row in summary_df.iloc[1:].iterrows():
            # if same seeds were used
            try:
                t_stat, p_val = stats.ttest_ind(
                    df[(df['param_name']==best['config']) | (df['param_value'].str.contains(best['config'][:20], na=False))][metric],
                    df[(df['param_name']==row['config']) | (df['param_value'].str.contains(row['config'][:20], na=False))][metric],
                    equal_var=False 
                )
                summary_df.loc[idx, 'p_vs_best'] = p_val
                summary_df.loc[idx, 'significant'] = p_val < 0.05
            except:
                pass
    
    return summary_df

