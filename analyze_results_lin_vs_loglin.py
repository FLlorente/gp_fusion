import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_results(filename, exclude_validation_proportion=None):
    df = pd.read_csv(filename)

    # Exclude specified validation_proportion values if any
    if exclude_validation_proportion is not None:
        df = df[~df['validation_proportion'].isin(exclude_validation_proportion)]

    # Get unique datasets
    unique_datasets = df['dataset'].unique()

    for dataset in unique_datasets:
        df_subset = df[df['dataset'] == dataset]

        # Melt the dataframe to have a long-form dataset suitable for seaborn
        df_melted = df_subset.melt(id_vars=['dataset', 'split', 'num_experts', 'validation_proportion'],
                            value_vars=['nlpd_single_gp', 'nlpd_experts', 'nlpd_phs', 'nlpd_bhs'],
                            var_name='method', value_name='nlpd')

        # Mapping for method names
        method_mapping = {
            'nlpd_single_gp': 'Full GP',
            'nlpd_experts': 'Experts',
            'nlpd_phs': 'PHS',
            'nlpd_bhs': 'BHS'
        }
        df_melted['method'] = df_melted['method'].map(method_mapping)

        # Create the plot
        g = sns.FacetGrid(df_melted, row='num_experts', col='validation_proportion', margin_titles=True, sharey=True)
        g.map_dataframe(sns.boxplot, x='method', y='nlpd', palette='Set3')

        # Adjust plot aesthetics
        g.set_axis_labels('Method', 'Negative Log Predictive Density (NLPD)')
        g.set_titles(row_template='{row_name} Experts', col_template='{col_name} Validation Proportion')
        plt.subplots_adjust(top=0.9)
        g.figure.suptitle(f'Comparison of NLPD Across Methods, Number of Experts, and Validation Proportion for {dataset} Dataset')
        
        # Save the plot for each dataset
        plt.savefig(f'comparison_nlpd_{dataset}.png')
        # plt.show()

if __name__ == '__main__':
    analyze_results('results_lin_vs_loglin.csv')
