import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(results_path):
    """Load all experimental results."""
    return pd.read_csv(results_path)

def statistical_comparison(df, metric='test_ap'):
    """Perform statistical tests between models."""
    results = {}
    
    # Group by model, dataset, eval_type, neg_strat
    grouped = df.groupby(['model', 'dataset', 'evaluation_type', 'negative_sampling_strategy'])
    
    for name, group in grouped:
        if len(group) >= 3:  # Need at least 3 seeds for stats
            mean_score = group[metric].mean()
            std_score = group[metric].std()
            ci_lower = mean_score - 1.96 * std_score / np.sqrt(len(group))
            ci_upper = mean_score + 1.96 * std_score / np.sqrt(len(group))
            
            results[name] = {
                'mean': mean_score,
                'std': std_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_seeds': len(group)
            }
    
    return results

def pairwise_ttest(df, model1, model2, metric='test_ap'):
    """Perform pairwise t-test between two models."""
    df1 = df[df['model'] == model1][metric]
    df2 = df[df['model'] == model2][metric]
    
    if len(df1) > 1 and len(df2) > 1:
        t_stat, p_value = stats.ttest_ind(df1, df2, equal_var=False)
        return {'t_statistic': t_stat, 'p_value': p_value, 'significant': p_value < 0.05}
    return None

def create_comparison_table(df, metric='test_ap'):
    """Create publication-ready comparison table."""
    # Pivot table: models x datasets
    pivot = df.pivot_table(
        values=metric,
        index=['model', 'negative_sampling_strategy'],
        columns='dataset',
        aggfunc='mean'
    )
    return pivot

def plot_performance_comparison(df, metric='test_ap'):
    """Create comprehensive performance comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model comparison across datasets
    sns.boxplot(data=df, x='dataset', y=metric, hue='model', ax=axes[0,0])
    axes[0,0].set_title(f'{metric.upper()} Comparison Across Datasets')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Negative sampling strategy impact
    sns.boxplot(data=df, x='negative_sampling_strategy', y=metric, hue='model', ax=axes[0,1])
    axes[0,1].set_title('Impact of Negative Sampling Strategy')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Evaluation type comparison
    sns.boxplot(data=df, x='evaluation_type', y=metric, hue='model', ax=axes[1,0])
    axes[1,0].set_title('Transductive vs Inductive Performance')
    
    # 4. Statistical significance heatmap
    models = df['model'].unique()
    datasets = df['dataset'].unique()
    p_values = np.zeros((len(models), len(models)))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i != j:
                result = pairwise_ttest(df[df['dataset'] == 'wikipedia'], m1, m2, metric)
                p_values[i, j] = result['p_value'] if result else 1.0
    
    sns.heatmap(p_values, xticklabels=models, yticklabels=models, 
                annot=True, cmap='coolwarm', ax=axes[1,1])
    axes[1,1].set_title('Statistical Significance (p-values)')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_negative_sampling_bias(df, metric='test_ap'):
    """Analyze H9: Negative Sampling Bias."""
    print("=== H9: Negative Sampling Bias Analysis ===")
    
    # Compare random vs historical vs inductive
    strategies = ['random', 'historical', 'inductive']
    for strategy in strategies:
        strat_data = df[df['negative_sampling_strategy'] == strategy]
        if not strat_data.empty:
            mean_ap = strat_data[metric].mean()
            std_ap = strat_data[metric].std()
            print(f"{strategy:12}: {mean_ap:.3f} ± {std_ap:.3f}")
    
    # Statistical test: random vs historical
    random_data = df[df['negative_sampling_strategy'] == 'random'][metric]
    historical_data = df[df['negative_sampling_strategy'] == 'historical'][metric]
    
    if len(random_data) > 1 and len(historical_data) > 1:
        t_stat, p_val = stats.ttest_ind(random_data, historical_data)
        print(f"\nRandom vs Historical: t={t_stat:.3f}, p={p_val:.3f}")
        if p_val < 0.05:
            print("✓ Significant difference confirmed (H9 supported)")
        else:
            print("✗ No significant difference found")

def main():
    results_path = Path('all_results.csv')
    if not results_path.exists():
        print("No results file found!")
        return
    
    df = load_results(results_path)
    print(f"Loaded {len(df)} experimental results")
    
    # Basic statistics
    stats_results = statistical_comparison(df)
    print("\n=== Performance Statistics ===")
    for config, metrics in stats_results.items():
        model, dataset, eval_type, neg_strat = config
        print(f"{model:10} | {dataset:10} | {eval_type:12} | {neg_strat:12} | "
              f"{metrics['mean']:.3f} ± {metrics['std']:.3f}")
    
    # Statistical analysis
    analyze_negative_sampling_bias(df)
    
    # Pairwise comparisons
    print("\n=== Pairwise Model Comparisons (Wikipedia) ===")
    wikipedia_data = df[df['dataset'] == 'wikipedia']
    models = wikipedia_data['model'].unique()
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                result = pairwise_ttest(wikipedia_data, m1, m2)
                if result:
                    print(f"{m1} vs {m2}: p={result['p_value']:.3f} "
                          f"{'***' if result['significant'] else ''}")
    
    # Create visualizations
    plot_performance_comparison(df)
    
    # Save detailed analysis
    comparison_table = create_comparison_table(df)
    comparison_table.to_csv('comparison_table.csv')
    print("\nDetailed comparison table saved to 'comparison_table.csv'")

if __name__ == "__main__":
    main()