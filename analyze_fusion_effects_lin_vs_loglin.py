import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to aggregate data by number of experts and Validation Proportion
def aggregate_data(df):
    results = []
    unique_datasets = df['dataset'].unique()
    for dataset in unique_datasets:
        df_subset = df[df['dataset'] == dataset]
        aggregation = df_subset.groupby(['num_experts', 'validation_proportion', 'method']).agg(
            mean_nlpd=('nlpd', 'mean')
        ).reset_index()
        aggregation['dataset'] = dataset
        results.append(aggregation)
    aggregated_df = pd.concat(results, ignore_index=True)
    return aggregated_df

# Function to calculate percentage difference with respect to Full GP
def calculate_percentage_difference(df, baseline_method='Full GP'):
    percentage_df = df.copy()
    unique_datasets = df['dataset'].unique()
    for dataset in unique_datasets:
        for num_experts in df['num_experts'].unique():
            baseline_nlpd = df[(df['dataset'] == dataset) & (df['method'] == baseline_method) & (df['num_experts'] == num_experts)]['mean_nlpd'].values
            if baseline_nlpd.size > 0:
                baseline_value = baseline_nlpd[0]
                percentage_df.loc[(percentage_df['dataset'] == dataset) & (percentage_df['num_experts'] == num_experts), 'mean_nlpd'] = \
                    (df.loc[(df['dataset'] == dataset) & (df['num_experts'] == num_experts), 'mean_nlpd'] - baseline_value) / abs(baseline_value) * 100
    return percentage_df

# Function to plot percentage difference (relative performance) for each dataset
def plot_percentage_difference_per_dataset(aggregated_df):
    unique_datasets = aggregated_df['dataset'].unique()
    for dataset in unique_datasets:
        df_subset = aggregated_df[aggregated_df['dataset'] == dataset]
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df_subset, x='num_experts', y='mean_nlpd', hue='method', marker='o')
        
        plt.title(f'Percentage Difference with Respect to Full GP for {dataset} Dataset')
        plt.xlabel('Number of Experts')
        plt.ylabel('Percentage Difference in Mean NLPD')
        # plt.yscale('log')
        plt.legend(title='Method')
        plt.grid(True)
        # plt.show()
        plt.savefig(f'nlpd_vs_num_experts_{dataset}.png')

# Function to plot aggregated percentage difference
def plot_aggregated_percentage_difference(aggregated_df, x_var, title_suffix):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=aggregated_df, x=x_var, y='mean_nlpd', hue='method', marker='o')
    
    plt.title(f'Aggregated Percentage Difference with Respect to Full GP {title_suffix}')
    plt.xlabel(x_var.replace('_', ' ').title())
    plt.ylabel('Percentage Difference in Mean NLPD')
    plt.legend(title='Method')
    # plt.ylim((-10,300))
    plt.grid(True)
    # plt.show()
    plt.savefig(f'agg_nlpd_vs_{x_var}.png')

# ---- Load the data ------ #
file_path = './results_lin_vs_loglin.csv'  


df = pd.read_csv(file_path)

# Exclude specific datasets (the ones where PHS diverged...)
exclude_datasets = ['forest', 'fertility', 'autos',]
df = df[~df['dataset'].isin(exclude_datasets)]

# Melt the dataframe to have a long-form dataset suitable for analysis
df_melted = df.melt(id_vars=['dataset', 'split', 'num_experts', 'validation_proportion'],
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

# Filter out Full GP from the analysis by number of experts, but include it for relative calculation
df_melted_experts_analysis = df_melted[df_melted['method'].isin(['BHS', 'PHS', 'Experts', 'Full GP'])]

# Aggregate the data
aggregated_data = aggregate_data(df_melted_experts_analysis)

# Calculate percentage difference
percentage_difference_df = calculate_percentage_difference(aggregated_data, baseline_method='Full GP')

# Filter out Full GP from the percentage difference plot
percentage_difference_df = percentage_difference_df[percentage_difference_df['method'] != 'Full GP']

# Plot the percentage difference for each dataset
# plot_percentage_difference_per_dataset(percentage_difference_df)

# Plot the aggregated percentage difference vs number of experts
plot_aggregated_percentage_difference(percentage_difference_df, 'num_experts', 'vs Number of Experts')

# Plot the aggregated percentage difference vs validation proportion
plot_aggregated_percentage_difference(percentage_difference_df, 'validation_proportion', 'vs Validation Proportion')
