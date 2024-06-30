# Differences wrt v0:   we do the num_experts loop with the full GP fixed
#                       we do the points_per_split loop with fixed experts

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.fusion_methods import compute_neg_log_like, product_fusion, train_and_predict_fusion_method
from modules.model_training import train_and_predict_single_gp, train_expert, store_predictions_for_experts
from modules.phs import phs
from modules.bhs import bhs

def run_experiment(config, resume=False):
    results = []
    datasets = config['datasets']
    parameters = config['parameters']
    fixed_params = config['fixed']

    # Load existing results if resuming
    if resume and os.path.exists('experiment_results.csv'):
        results = pd.read_csv('experiment_results.csv').to_dict('records')
        completed_runs = {(r['dataset'], r['split'], r['num_experts'], r['points_per_split']) for r in results}
    else:
        completed_runs = set()

    total_runs = len(datasets) * len(parameters['num_experts']) * len(parameters['points_per_split']) * max(d['splits'] for d in datasets)
    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        for dataset in datasets:
            for split in range(dataset['splits']):
                # Load and normalize data
                X_train, y_train, X_test, y_test = load_and_normalize_data(dataset['name'], split)

                # Compute single GP using all training data
                test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, fixed_params['kappa'], fixed_params['lambdaa'])
                nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)

                for num_experts in parameters['num_experts']:
                    # Split dataset for experts
                    n_data_per_expert = X_train.shape[0] // num_experts    # equal number of observations per expert (last split contains the remaining observations pairs)
                    splits = split_dataset(X_train, y_train, n_splits=num_experts, split_size=n_data_per_expert, with_replacement=False)

                    # Train experts and store models
                    experts = []
                    for X_split, y_split in splits:
                        model, likelihood = train_expert(X_split, y_split, fixed_params['kappa'], fixed_params['lambdaa'])
                        experts.append((model, likelihood))

                    # Store predictions for experts on the test set
                    mu_preds_test, std_preds_test = store_predictions_for_experts(experts, X_test)

                    # Compute negative log likelihood for experts
                    nlpd_experts = compute_neg_log_like(mu_preds_test, std_preds_test, y_test)

                    # Compute GPOE
                    mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test, std_preds_test, splits, fixed_params['kappa'])
                    nlpd_gpoe = compute_neg_log_like(mus_gpoe, stds_gpoe, y_test)

                    for points_per_split in parameters['points_per_split']:
                        if (dataset['name'], split, num_experts, points_per_split) in completed_runs:
                            pbar.update(1)
                            continue

                        pbar.set_postfix({
                            'dataset': dataset['name'],
                            'split': split,
                            'num_experts': num_experts,
                            'points_per_split': points_per_split
                        })

                        try:
                            # Create validation set
                            X_val, y_val = create_validation_set(splits, points_per_split)

                            # Store predictions for experts on the validation set
                            mu_preds_val, std_preds_val = store_predictions_for_experts(experts, X_val)

                            # PHS training and prediction
                            preds_phs, lpd_phs_test = train_and_predict_fusion_method(
                                model=phs,
                                X_val=X_val,
                                mu_preds_val=mu_preds_val,
                                std_preds_val=std_preds_val,
                                y_val=y_val,
                                X_test=X_test,
                                mu_preds_test=mu_preds_test,
                                std_preds_test=std_preds_test,
                                y_test=y_test
                            )

                            # BHS training and prediction
                            preds_bhs, lpd_bhs_test = train_and_predict_fusion_method(
                                model=bhs,
                                X_val=X_val,
                                mu_preds_val=mu_preds_val,
                                std_preds_val=std_preds_val,
                                y_val=y_val,
                                X_test=X_test,
                                mu_preds_test=mu_preds_test,
                                std_preds_test=std_preds_test,
                                y_test=y_test
                            )

                            # Collect results
                            result = {
                                'dataset': dataset['name'],
                                'split': split,
                                'num_experts': num_experts,
                                'points_per_split': points_per_split,
                                'nlpd_single_gp': nlpd_single_gp.mean(),  # .mean() is not necessary 
                                'nlpd_experts': nlpd_experts.mean(),  # we store the average nlpd of the experts
                                'nlpd_gpoe': nlpd_gpoe.mean(),   # .mean() is not necessary 
                                'nlpd_phs': -lpd_phs_test.mean(), 
                                'nlpd_bhs': -lpd_bhs_test.mean()
                            }
                            results.append(result)

                            # Save intermediate results
                            df = pd.DataFrame(results)
                            df.to_csv('experiment_results.csv', index=False)

                            pbar.update(1)

                        except Exception as e:
                            print(f"Error running experiment for dataset={dataset['name']}, split={split}, num_experts, {num_experts}, points_per_split={points_per_split}: {e}")
                            continue

    return results

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = run_experiment(config, resume=True)
    # Save final results
    save_results(results, 'experiment_results.csv')

if __name__ == '__main__':
    main()
