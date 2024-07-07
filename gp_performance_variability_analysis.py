from modules.data_handling import load_and_normalize_data
from modules.fusion_methods import compute_neg_log_like
from modules.model_training import train_and_predict_single_gp

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import multiprocessing as mp
import os
import traceback

# Parameters
# datasets = [
#     'autompg', 'concreteslump', 'energy', 'forest', 'solar', 'stock', 'yacht',
#     'airfoil', 'autos', 'breastcancer', 'concrete', 'housing', 'machine',
#     'pendulum', 'servo', 'wine'
# ]
datasets = ['energy', 'solar', 'yacht']  # estos faltan....

configs = [  
    # Low Noise, Short Lengthscale Regime
    {"kappa": 50, "lambdaa": 10, "lr": 0.1, "training_iter": 100},
    {"kappa": 50, "lambdaa": 10, "lr": 0.01, "training_iter": 500},
    {"kappa": 50, "lambdaa": 8, "lr": 0.1, "training_iter": 100},
    {"kappa": 50, "lambdaa": 8, "lr": 0.01, "training_iter": 500},
    
    # High Noise, Long Lengthscale Regime
    {"kappa": 2, "lambdaa": 1, "lr": 0.1, "training_iter": 100},
    {"kappa": 2, "lambdaa": 1, "lr": 0.01, "training_iter": 500},
    {"kappa": 5, "lambdaa": 2, "lr": 0.1, "training_iter": 100},
    {"kappa": 5, "lambdaa": 2, "lr": 0.01, "training_iter": 500},
    
    # Balanced Regime
    {"kappa": 10, "lambdaa": 5, "lr": 0.1, "training_iter": 100},
    {"kappa": 10, "lambdaa": 5, "lr": 0.01, "training_iter": 500},
    {"kappa": 20, "lambdaa": 5, "lr": 0.1, "training_iter": 100},
    {"kappa": 20, "lambdaa": 5, "lr": 0.01, "training_iter": 500},
]

num_seeds = 10
metric = "nlpd"  # Change to "mse" for MSE comparison

output_dir = "variability_gp_small_datasets"
os.makedirs(output_dir, exist_ok=True)

def compute_metric(test_preds, y_test, metric):
    if metric == "nlpd":
        return compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), 
                                    np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), 
                                    y_test)
    elif metric == "mse":
        return np.mean((test_preds.mean.numpy() - y_test) ** 2)
    else:
        raise ValueError("Unknown metric")

def analyze_dataset(dataset_name, metric):
    results = []
    progress_bar = tqdm(total=len(configs) * num_seeds, desc=f"Progress for {dataset_name}", unit="iteration")
    
    for config in configs:
        metric_values = []
        errors = []
        for split in range(num_seeds):
            try:    
                X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name, split, normalize_y=True)
                
                test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, 
                                                            config['kappa'], 
                                                            config['lambdaa'],
                                                            lr=config['lr'],
                                                            training_iter=config['training_iter'])
                
                metric_value = compute_metric(test_preds, y_test, metric)
                metric_values.append(metric_value)
            except Exception as e:
                error_message = f"Error in dataset {dataset_name}, config {config}, split {split}: {str(e)}"
                errors.append(error_message)
                traceback.print_exc()    
            progress_bar.update(1)

        results.append({
            "config": config,
            "metric_values": metric_values
        })

    progress_bar.close()

    # Prepare data for statistical testing
    all_metric_values = []
    group_labels = []
    for idx, result in enumerate(results):
        all_metric_values.extend(result["metric_values"])
        group_labels.extend([idx + 1] * len(result["metric_values"]))  # Use numerical labels

    # Perform one-way ANOVA test
    f_val, p_val = stats.f_oneway(*[result["metric_values"] for result in results])
    summary = f"F-value for {dataset_name}: {f_val}, p-value: {p_val}\n"

    if p_val < 0.05:
        # Perform Tukey's HSD post-hoc test
        tukey_result = pairwise_tukeyhsd(endog=np.array(all_metric_values), groups=np.array(group_labels), alpha=0.05)
        summary += str(tukey_result) + "\n"

        # Plot the results
        fig, ax = plt.subplots()
        tukey_result.plot_simultaneous(ax=ax)
        ax.set_title(f"Tukey HSD Test Results for {dataset_name} ({metric.upper()})")
        plt.savefig(os.path.join(output_dir, f"tukey_results_{dataset_name}_{metric}.png"))
        plt.close(fig)

    # Convert lists to Pandas DataFrame for plotting
    df = pd.DataFrame({
        'Configuration': np.stack(group_labels,axis=0).squeeze(),
        'Metric': np.stack(all_metric_values,axis=0).squeeze()
    })

    # Ensure 'Configuration' is treated as a categorical variable with explicit order
    df['Configuration'] = pd.Categorical(df['Configuration'], categories=sorted(set(group_labels)), ordered=True)

    # Convert numerical labels to string labels for plotting
    df['Configuration'] = df['Configuration'].apply(lambda x: f"Config {x}")

    # Display results
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x='Configuration', y='Metric', data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title(f"{metric.upper()} Across Different Configurations for {dataset_name}")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Configuration")
    plt.savefig(os.path.join(output_dir, f"boxplot_{dataset_name}_{metric}.png"))
    plt.close(fig)

    return summary

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        summaries = pool.starmap(analyze_dataset, [(dataset, metric) for dataset in datasets])

    with open(os.path.join(output_dir, f"summary_results_{metric}.txt"), "w") as summary_file:
        for summary in summaries:
            summary_file.write(summary)
            summary_file.write("\n")
