import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed




from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.fusion_methods import compute_neg_log_like, train_and_predict_fusion_method
from modules.model_training import train_and_predict_single_gp, train_expert, store_predictions_for_experts
from modules.phs import phs
from modules.bhs import bhs


def run_single_experiment(dataset, split, num_experts, validation_proportion, fixed_params):
    try:

        # Load and normalize data
        X_train, y_train, X_test, y_test = load_and_normalize_data(dataset['name'], 
                                                                   split,
                                                                   normalize_x_method="z-score",
                                                                   normalize_y=True)

        # Compute single GP using all training data
        test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, fixed_params['kappa'], fixed_params['lambdaa'])
        nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)


        # Partition the X_train into X_train_train (for training the experts) and X_val (for training the weights)
        np.random.seed(11)
        num_val_samples = int(validation_proportion*len(X_train))

        # Shuffle the data indices
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]

        X_train_train = X_train[train_indices]
        y_train_train = y_train[train_indices]

        X_val = X_train[val_indices]
        y_val = y_train[val_indices]


        # Partition dataset for experts
        splits = split_dataset(X_train_train, y_train_train, n_splits=num_experts, with_replacement=False)

        # Train experts and store models
        experts = []
        for X_split, y_split in splits:
            model, likelihood = train_expert(X_split, y_split, fixed_params['kappa'], fixed_params['lambdaa'])
            experts.append((model, likelihood))

        # Store predictions for experts on the test set
        mu_preds_test, std_preds_test, std_preds_prior_test = store_predictions_for_experts(experts, X_test)

        # Compute negative log likelihood for experts
        nlpd_experts = compute_neg_log_like(mu_preds_test, std_preds_test, y_test)


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
            'validation_proportion': validation_proportion,
            'nlpd_single_gp': nlpd_single_gp.mean(),
            'nlpd_experts': nlpd_experts.mean(),
            'nlpd_phs': -lpd_phs_test.mean(),
            'nlpd_bhs': -lpd_bhs_test.mean()
        }
        return result

    except Exception as e:
        print(f"Error running experiment for dataset={dataset['name']}, split={split}, num_experts={num_experts}, validation_proportion={validation_proportion}: {e}")
        return None

def run_experiment(config, resume=False):
    results = []
    datasets = config['datasets']
    parameters = config['parameters']
    fixed_params = config['fixed']

    # Load existing results if resuming
    if resume and os.path.exists('results_lin_vs_loglin.csv'):
        results = pd.read_csv('results_lin_vs_loglin.csv').to_dict('records')
        completed_runs = {(r['dataset'], r['split'], r['num_experts'], r['validation_proportion']) for r in results}
    else:
        completed_runs = set()

    total_runs = len(datasets) * len(parameters['num_experts']) * len(parameters['validation_proportion']) * max(d['splits'] for d in datasets)
    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        # with ThreadPoolExecutor() as executor:
        with ProcessPoolExecutor() as executor:    
            futures = []
            for dataset in datasets:
                for split in range(dataset['splits']):
                    for num_experts in parameters['num_experts']:
                        for validation_proportion in parameters['validation_proportion']:
                            if (dataset['name'], split, num_experts, validation_proportion) in completed_runs:
                                pbar.update(1)
                                continue
                            futures.append(executor.submit(run_single_experiment, dataset, split, num_experts, validation_proportion, fixed_params))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                    # Save intermediate results
                    df = pd.DataFrame(results)
                    df.to_csv('results_lin_vs_loglin.csv', index=False)
                pbar.update(1)
    
    return results

def save_results(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def main():
    with open('config_lin_vs_loglin.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    results = run_experiment(config, resume=True)
    save_results(results, 'results_lin_vs_loglin.csv')

if __name__ == '__main__':
    main()
