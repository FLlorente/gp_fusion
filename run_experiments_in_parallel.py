import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed


from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.fusion_methods import compute_neg_log_like, product_fusion, train_and_predict_fusion_method
from modules.model_training import train_and_predict_single_gp, train_expert, store_predictions_for_experts
from modules.phs import phs
from modules.bhs import bhs


def run_single_experiment(dataset, split, num_experts, points_per_split, fixed_params):
    try:

        # import jax
        # import jax.numpy as jnp
        # import numpyro
        # import numpyro.distributions as dist
        # from numpyro.infer import MCMC, NUTS

        # # Ensure numpyro is initialized properly for each process
        # jax.config.update("jax_platform_name", "cpu")
        # numpyro.set_platform("cpu")
        # numpyro.set_host_device_count(1)    

        # Load and normalize data
        X_train, y_train, X_test, y_test = load_and_normalize_data(dataset['name'], split)

        # Compute single GP using all training data
        test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, fixed_params['kappa'], fixed_params['lambdaa'])
        nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)

        # Split dataset for experts
        n_data_per_expert = X_train.shape[0] // num_experts
        splits = split_dataset(X_train, y_train, n_splits=num_experts, split_size=n_data_per_expert, with_replacement=False)

        # Train experts and store models
        experts = []
        for X_split, y_split in splits:
            model, likelihood = train_expert(X_split, y_split, fixed_params['kappa'], fixed_params['lambdaa'])
            experts.append((model, likelihood))

        # Store predictions for experts on the test set
        mu_preds_test, std_preds_test, std_preds_prior_test = store_predictions_for_experts(experts, X_test)

        # Compute negative log likelihood for experts
        nlpd_experts = compute_neg_log_like(mu_preds_test, std_preds_test, y_test)

        # Compute GPOE
        mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test, std_preds_test, std_preds_prior_test)
        nlpd_gpoe = compute_neg_log_like(mus_gpoe, stds_gpoe, y_test)

        # Create validation set
        X_val, y_val = create_validation_set(splits, points_per_split)

        # Store predictions for experts on the validation set
        mu_preds_val, std_preds_val, _ = store_predictions_for_experts(experts, X_val) # we don't need the prior predictive variances here...

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
            'nlpd_single_gp': nlpd_single_gp.mean(),
            'nlpd_experts': nlpd_experts.mean(),
            'nlpd_gpoe': nlpd_gpoe.mean(),
            'nlpd_phs': -lpd_phs_test.mean(),
            'nlpd_bhs': -lpd_bhs_test.mean()
        }
        return result

    except Exception as e:
        print(f"Error running experiment for dataset={dataset['name']}, split={split}, num_experts={num_experts}, points_per_split={points_per_split}: {e}")
        return None

def run_experiment(config, resume=False):
    results = []
    datasets = config['datasets']
    parameters = config['parameters']
    fixed_params = config['fixed']

    # Load existing results if resuming
    if resume and os.path.exists('experiment_results_parallel.csv'):
        results = pd.read_csv('experiment_results_parallel.csv').to_dict('records')
        completed_runs = {(r['dataset'], r['split'], r['num_experts'], r['points_per_split']) for r in results}
    else:
        completed_runs = set()

    total_runs = len(datasets) * len(parameters['num_experts']) * len(parameters['points_per_split']) * max(d['splits'] for d in datasets)
    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        # with ThreadPoolExecutor() as executor:
        with ProcessPoolExecutor() as executor:    
            futures = []
            for dataset in datasets:
                for split in range(dataset['splits']):
                    for num_experts in parameters['num_experts']:
                        for points_per_split in parameters['points_per_split']:
                            if (dataset['name'], split, num_experts, points_per_split) in completed_runs:
                                pbar.update(1)
                                continue
                            futures.append(executor.submit(run_single_experiment, dataset, split, num_experts, points_per_split, fixed_params))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                    # Save intermediate results
                    df = pd.DataFrame(results)
                    df.to_csv('experiment_results_parallel.csv', index=False)
                pbar.update(1)
    
    return results

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_experiment(config, resume=True)
    save_results(results, 'experiment_results_parallel.csv')

if __name__ == '__main__':
    main()
