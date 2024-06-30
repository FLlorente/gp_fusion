import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.prediction_storage import store_predictions
from modules.fusion_methods import compute_neg_log_like, product_fusion, phs, bhs, train_stacking, predict_stacking
from modules.model_training import train_and_predict_single_gp

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
            for num_experts in parameters['num_experts']:
                for points_per_split in parameters['points_per_split']:
                    for split in range(dataset['splits']):
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
                            # Load and normalize data
                            X_train, y_train, X_test, y_test = load_and_normalize_data(dataset['name'], split)

                            # Split dataset and create validation set
                            n_data_per_expert = X_train.shape[0] // num_experts
                            splits = split_dataset(X_train, y_train, n_splits=num_experts, split_size=n_data_per_expert, with_replacement=False)
                            X_val, y_val = create_validation_set(splits, points_per_split)

                            # Store predictions
                            mu_preds_val, std_preds_val, mu_preds_test, std_preds_test = store_predictions(splits, X_val, X_test, fixed_params['kappa'], fixed_params['lambdaa'])

                            # Compute negative log likelihood for experts
                            nlpd_experts = compute_neg_log_like(mu_preds_test, std_preds_test, y_test)

                            # Single GP using all training data
                            test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_val, fixed_params['kappa'], fixed_params['lambdaa'])
                            nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)

                            # Fusion using gpoe
                            mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test, std_preds_test, y_train, fixed_params['kappa'])
                            nlpd_gpoe = compute_neg_log_like(mus_gpoe, stds_gpoe, y_test)

                            # PHS training and prediction
                            samples_phs = train_stacking(
                                model=phs,
                                X_val=X_val,
                                mu_preds_val=mu_preds_val,
                                std_preds_val=std_preds_val,
                                y_val=y_val,
                            )

                            preds_phs, lpd_phs_test = predict_stacking(
                                model=phs,
                                samples=samples_phs,
                                X_val=X_val,
                                X_test=X_test,
                                mu_preds_test=mu_preds_test,
                                std_preds_test=std_preds_test,
                                y_test=y_test,
                                prior_mean=lambda x: -np.log(mu_preds_test.shape[1]) * np.ones(x.shape[0]),
                            )

                            # BHS training and prediction
                            samples_bhs = train_stacking(
                                model=bhs,
                                X_val=X_val,
                                mu_preds_val=mu_preds_val,
                                std_preds_val=std_preds_val,
                                y_val=y_val,
                            )

                            preds_bhs, lpd_bhs_test = predict_stacking(
                                model=bhs,
                                samples=samples_bhs,
                                X_val=X_val,
                                X_test=X_test,
                                mu_preds_test=mu_preds_test,
                                std_preds_test=std_preds_test,
                                y_test=y_test,
                                prior_mean=lambda x: -np.log(mu_preds_test.shape[1]) * np.ones(x.shape[0]),
                            )

                            # Collect results
                            result = {
                                'dataset': dataset['name'],
                                'split': split,
                                'num_experts': num_experts,
                                'points_per_split': points_per_split,
                                'nlpd_experts': nlpd_experts.mean(),      # average nlpd of the experts
                                'nlpd_single_gp': nlpd_single_gp.mean(),  # mean here doesn't do anything
                                'nlpd_gpoe': nlpd_gpoe.mean(),            # mean here doesn't do anything
                                'nlpd_phs': -lpd_phs_test.mean(),
                                'nlpd_bhs': -lpd_bhs_test.mean()
                            }
                            results.append(result)

                            # Save intermediate results
                            df = pd.DataFrame(results)
                            df.to_csv('experiment_results.csv', index=False)

                            pbar.update(1)

                        except Exception as e:
                            print(f"Error running experiment for dataset={dataset['name']}, split={split}, num_experts={num_experts}, points_per_split={points_per_split}: {e}")
                            continue

    return results

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = run_experiment(config, resume=True)
    save_results(results, 'experiment_results.csv')

if __name__ == '__main__':
    main()
