# plot and charts
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