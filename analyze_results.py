import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_results(filename, exclude_points_per_split=None):
    df = pd.read_csv(filename)

    # Exclude specified points_per_split values if any
    if exclude_points_per_split is not None:
        df = df[~df['points_per_split'].isin(exclude_points_per_split)]

    # Get unique datasets
    unique_datasets = df['dataset'].unique()

    for dataset in unique_datasets:
        df_subset = df[df['dataset'] == dataset]

        # Melt the dataframe to have a long-form dataset suitable for seaborn
        df_melted = df_subset.melt(id_vars=['dataset', 'split', 'num_experts', 'points_per_split'],
                            value_vars=['nlpd_single_gp', 'nlpd_experts', 'nlpd_gpoe', 'nlpd_phs', 'nlpd_bhs'],
                            var_name='method', value_name='nlpd')

        # Mapping for method names
        method_mapping = {
            'nlpd_single_gp': 'Full GP',
            'nlpd_experts': 'Experts',
            'nlpd_gpoe': 'gPoE',
            'nlpd_phs': 'PHS',
            'nlpd_bhs': 'BHS'
        }
        df_melted['method'] = df_melted['method'].map(method_mapping)

        # Create the plot
        g = sns.FacetGrid(df_melted, row='num_experts', col='points_per_split', margin_titles=True, sharey=True)
        g.map_dataframe(sns.boxplot, x='method', y='nlpd', palette='Set3')

        # Adjust plot aesthetics
        g.set_axis_labels('Method', 'Negative Log Predictive Density (NLPD)')
        g.set_titles(row_template='{row_name} Experts', col_template='{col_name} Points per Split')
        plt.subplots_adjust(top=0.9)
        g.figure.suptitle(f'Comparison of NLPD Across Methods, Number of Experts, and Points per Split for {dataset} Dataset')
        
        # Save the plot for each dataset
        # plt.savefig(f'comparison_nlpd_{dataset}.png')
        plt.show()

if __name__ == '__main__':
    # analyze_results('experiment_results.csv', exclude_points_per_split=[20])
    analyze_results('experiment_results_parallel.csv', exclude_points_per_split=[20])
