
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


# Parameters
datasets = ['autompg', 'concreteslump', 'energy', 'forest', 'solar', 'stock', 'yacht',
            'airfoil', 'autos', 'breastcancer', 'concrete', 'housing', 'machine', 
            'pendulum', 'servo', 'wine']

dataset_name = 'concrete'


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
results = []
progress_bar = tqdm(total=len(configs) * num_seeds, desc="Progress", unit="iteration")

for config in configs:
    nlpd_gp = []
    for split in range(num_seeds):
        X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name, split, normalize_y=True)
        
        test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, 
                                                    config['kappa'], 
                                                    config['lambdaa'],
                                                    lr=config['lr'],
                                                    training_iter=config['training_iter'])
        
        nlpd = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), 
                                    np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), 
                                    y_test)
        nlpd_gp.append(nlpd)
        progress_bar.update(1)

    results.append({
        "config": config,
        "nlpd_values": nlpd_gp
    })

progress_bar.close()

# Prepare data for statistical testing
all_nlpd_values = []
group_labels = []
for idx, result in enumerate(results):
    all_nlpd_values.extend(result["nlpd_values"])
    group_labels.extend([idx + 1] * len(result["nlpd_values"]))  # Use numerical labels

# Convert lists to Pandas DataFrame for plotting
df = pd.DataFrame({
    'Configuration': np.stack(group_labels, axis=0).squeeze(),
    'NLPD': np.stack(all_nlpd_values, axis=0).squeeze()
})

# Ensure 'Configuration' is treated as a categorical variable with explicit order
df['Configuration'] = pd.Categorical(df['Configuration'], categories=sorted(set(group_labels)), ordered=True)

# Perform one-way ANOVA test
f_val, p_val = stats.f_oneway(*[result["nlpd_values"] for result in results])
print(f"F-value: {f_val}, p-value: {p_val}")

if p_val < 0.05:
    # Perform Tukey's HSD post-hoc test
    tukey_result = pairwise_tukeyhsd(endog=np.array(all_nlpd_values), groups=np.array(group_labels), alpha=0.05)
    print(tukey_result)

    # Plot the results
    tukey_result.plot_simultaneous()
    plt.title("Tukey HSD Test Results")
    plt.show()

# Convert numerical labels to string labels for plotting
df['Configuration'] = df['Configuration'].apply(lambda x: f"Config {x}")

# Display results
plt.figure(figsize=(14, 7))
sns.boxplot(x='Configuration', y='NLPD', data=df)
plt.xticks(rotation=45)
plt.title("NLPD Across Different Configurations")
plt.ylabel("NLPD")
plt.xlabel("Configuration")
plt.show()