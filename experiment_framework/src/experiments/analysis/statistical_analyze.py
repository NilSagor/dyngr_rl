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

# def _save_and_plot(self, df: pd.DataFrame, param_name: str):
#         if df.empty:
#             return
        
#         csv_path = self.output_dir / f"{param_name.replace('.', '_')}_results.csv"
#         df.to_csv(csv_path, index=False)
        
#         plot_df = df.dropna(subset=['test_ap'])
#         if plot_df.empty:
#             return
        
#         try:
#             plot_df['param_num'] = pd.to_numeric(plot_df['param_value'])
#             agg = plot_df.groupby('param_num')['test_ap'].agg(['mean', 'std']).reset_index()
#             x_col = 'param_num'
#         except:
#             agg = plot_df.groupby('param_value')['test_ap'].agg(['mean', 'std']).reset_index()
#             x_col = 'param_value'
        
#         plt.figure(figsize=(10, 6))
#         plt.errorbar(agg[x_col], agg['mean'], yerr=agg['std'], marker='o', capsize=5)
#         plt.fill_between(agg[x_col], agg['mean'] - agg['std'], agg['mean'] + agg['std'], alpha=0.2)
#         plt.title(f"Sensitivity: {param_name}")
#         plt.xlabel(param_name)
#         plt.ylabel("Test AP")
#         plt.grid(True, alpha=0.3)
        
#         plot_path = self.output_dir / f"{param_name.replace('.', '_')}_plot.png"
#         plt.savefig(plot_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         logger.info(f"Plot saved: {plot_path}")