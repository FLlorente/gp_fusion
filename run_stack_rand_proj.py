import warnings
warnings.filterwarnings("ignore", message=".*omp.h header is not in the path, disabling OpenMP.*")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from uci_datasets import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

from modules.data_handling import load_data
from modules.model_training import train_and_predict_single_gp, train_and_predict_batched_gp
from modules.fusion_methods import product_fusion, compute_neg_log_like

from gpytorch.means import ZeroMean
from gpytorch.kernels import AdditiveStructureKernel, RBFKernel, ScaleKernel


def save_results(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def load_results(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def run_single_gp(dataset_name, gam=False, lr=0.1, training_iter=100):
    nlpd = []
    rmse = []
    for i in range(10):
        try:
            X_train, y_train, X_test, y_test, y_std = load_data(dataset_name, i)

            if gam:
                kernel = AdditiveStructureKernel(base_kernel=ScaleKernel(RBFKernel()), 
                                                 num_dims=X_train.shape[1])
            else:
                kernel = None

            test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test,
                                                        # kappa=2,
                                                        # lambdaa=2,
                                                        mean=ZeroMean(),
                                                        kernel=kernel,
                                                        lr=lr,
                                                        training_iter=training_iter,
                                                        initialiaze_hyper=False,  # if false, kappa and lambdaa don't matter!
                                                        )
            nlpd_now = compute_neg_log_like(test_preds.mean, np.sqrt(test_preds.variance), y_test)
            rmse_now = np.sqrt(np.mean((test_preds.mean.numpy().squeeze() - y_test) ** 2))

            nlpd.append(nlpd_now.squeeze())
            rmse.append(rmse_now)
        except Exception as e:
            print(f"There was an error during hyperparameter learning: {e}")

    nlpd = np.array(nlpd)
    rmse = np.array(rmse)

    return nlpd, rmse


def run_stacked_proj_gp_batched(dataset_name, num_projections, project_dim,
                                proj="random", lr=0.1, training_iter=100,
                                kernel=None):
    results = defaultdict(lambda: {"nlpd": [], "rmse": []})

    configs = [
        {'weighting': 'uniform', 'method': 'gPoE', 'normalize': True},
        {'weighting': 'entropy', 'method': 'gPoE', 'normalize': True},
    ]

    for split in range(10):
        X_train, y_train, X_test, y_test, y_std = load_data(dataset_name, split)

        mean_experts = []
        std_experts = []
        std_experts_prior = []

        X_train_batched = np.zeros((num_projections, X_train.shape[0], project_dim))
        X_test_batched = np.zeros((num_projections, X_test.shape[0], project_dim))
        y_train_batched = np.zeros((num_projections, y_train.shape[0]))
        for d in range(num_projections):
            if proj == "random":
                np.random.seed(d)
                P_proj = np.random.randn(X_train.shape[1], project_dim) / np.sqrt(project_dim)
            elif proj == "axis":
                P_proj = np.array([1.0 if i == d else 0.0 for i in range(X_train.shape[1])])
                P_proj = P_proj.reshape(-1, 1)

            X_train_batched[d, :, :] = np.matmul(X_train, P_proj)
            y_train_batched[d, :] = y_train
            X_test_batched[d, :, :] = np.matmul(X_test, P_proj)

        preds, preds_prior = train_and_predict_batched_gp(X_train_batched,
                                                          y_train_batched,
                                                          X_test_batched,
                                                          training_iter=training_iter, lr=lr,
                                                          kernel=kernel,
                                                          )

        mean_experts = preds.mean.numpy().T
        std_experts = np.sqrt(preds.variance.numpy().T)
        std_experts_prior = np.sqrt(preds_prior.variance.numpy().T)

        for config in configs:
            mean_fused, std_fused, _ = product_fusion(mean_experts,
                                                      std_experts,
                                                      std_experts_prior,
                                                      weighting=config["weighting"],
                                                      normalize=config["normalize"],
                                                      method=config["method"])

            nlpd_now = compute_neg_log_like(mean_fused, std_fused, y_test).squeeze()
            rmse_now = np.sqrt(np.mean((mean_fused.squeeze() - y_test.squeeze()) ** 2))

            config_key = f'{config["method"]}_{config["weighting"]}_normalize_{config["normalize"]}'
            results[config_key]["nlpd"].append(nlpd_now)
            results[config_key]["rmse"].append(rmse_now)

    mean_results = {key: {"mean_nlpd": np.mean(value["nlpd"]),
                          "mean_rmse": np.mean(value["rmse"]),
                          "std_err_nlpd": np.std(value["nlpd"]) / np.sqrt(10),
                          "std_err_rmse": np.std(value["rmse"]) / np.sqrt(10),
                          }
                    for key, value in results.items()}

    return mean_results


def run_experiment_for_dataset(dataset_name, num_proj_dim_vals, num_projections_vals, save_path):
    dataset_path = os.path.join(save_path, f"{dataset_name}_results.pkl")

    if not os.path.exists(dataset_path):

        print(f"Running experiments for {dataset_name}...")

        data = Dataset(dataset_name)
        X_train = data.x
        proj_dim_vals = np.linspace(1, X_train.shape[1], num_proj_dim_vals).astype(int)

        results = {
            "single_gp": run_single_gp(dataset_name),
            "gam_gp": run_single_gp(dataset_name, gam=True),
            "stacked_proj_gp_batched": []
        }

        for num_projections in num_projections_vals:
            nlpd = []
            rmse = []
            std_err_nlpd = []
            std_err_rmse = []
            for project_dim in proj_dim_vals:
                result = run_stacked_proj_gp_batched(dataset_name, num_projections, project_dim=project_dim)
                # substitute with the results of other weighting strategies, e.g., 
                # result["gPoE_entropy_normalize_True"][
                nlpd.append(result["gPoE_uniform_normalize_True"]["mean_nlpd"])
                rmse.append(result["gPoE_uniform_normalize_True"]["mean_rmse"])
                std_err_nlpd.append(result["gPoE_uniform_normalize_True"]["std_err_nlpd"])
                std_err_rmse.append(result["gPoE_uniform_normalize_True"]["std_err_rmse"])

            results["stacked_proj_gp_batched"].append({
                "num_projections": num_projections,
                "proj_dim_vals": proj_dim_vals,
                "nlpd": np.array(nlpd),
                "rmse": np.array(rmse),
                "std_err_nlpd": np.array(std_err_nlpd),
                "std_err_rmse": np.array(std_err_rmse)
            })

        save_results(results, dataset_path)
        print(f"Results for {dataset_name} saved.")
    else:
        print(f"Results for {dataset_name} already exist.")


def run_and_store_all_datasets(dataset_names, num_proj_dim_vals, num_projections_vals, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_experiment_for_dataset, dataset_name, num_proj_dim_vals, num_projections_vals, save_path): dataset_name
            for dataset_name in dataset_names
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            dataset_name = futures[future]
            try:
                future.result()
                print(f"Completed experiments for {dataset_name}")
            except Exception as e:
                print(f"Error running experiments for {dataset_name}: {e}")


def plot_results(dataset_names, num_proj_dim_vals, save_path):
    cmap = plt.get_cmap("tab10")

    for dataset_name in dataset_names:
        dataset_path = os.path.join(save_path, f"{dataset_name}_results.pkl")
        if not os.path.exists(dataset_path):
            print(f"Results for {dataset_name} do not exist. Please run the simulations first.")
            continue

        results = load_results(dataset_path)

        plt.figure(figsize=(12, 5))

        data = Dataset(dataset_name)
        proj_dim_vals = np.linspace(1, data.x.shape[1], num_proj_dim_vals).astype(int)

        # Plot RMSE
        plt.subplot(1, 2, 1)
        plt.axhline(results["single_gp"][1].mean(), label="ard RBF", color=cmap(0))
        # plt.axhline(results["single_gp"][1].mean() - results["single_gp"][1].std() / np.sqrt(10), color=cmap(0), linestyle='dashed')
        # plt.axhline(results["single_gp"][1].mean() + results["single_gp"][1].std() / np.sqrt(10), color=cmap(0), linestyle='dashed')
        plt.axhline(results["gam_gp"][1].mean(), label="gam RBF", color=cmap(1))
        # plt.axhline(results["gam_gp"][1].mean() - results["gam_gp"][1].std() / np.sqrt(10), color=cmap(1), linestyle='dashed')
        # plt.axhline(results["gam_gp"][1].mean() + results["gam_gp"][1].std() / np.sqrt(10), color=cmap(1), linestyle='dashed')

        for i, stacked_result in enumerate(results["stacked_proj_gp_batched"]):
            plt.plot(stacked_result["proj_dim_vals"], stacked_result["rmse"], label=f"stacked {stacked_result['num_projections']} random-proj", color=cmap(2 + i % 8))

        plt.ylabel("RMSE")
        plt.xlabel("Proj. dim")
        plt.legend()
        plt.title(f"RMSE for {dataset_name}")

        # Plot NLPD
        plt.subplot(1, 2, 2)
        plt.axhline(results["single_gp"][0].mean(), label="ard RBF", color=cmap(0))
        plt.axhline(results["gam_gp"][0].mean(), label="gam RBF", color=cmap(1))
        for i, stacked_result in enumerate(results["stacked_proj_gp_batched"]):
            plt.plot(stacked_result["proj_dim_vals"], stacked_result["nlpd"], label=f"stacked {stacked_result['num_projections']} random-proj", color=cmap(2 + i % 8))

        plt.ylabel("NLPD")
        plt.xlabel("Proj. dim")
        plt.legend()
        plt.title(f"NLPD for {dataset_name}")

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(save_path, f"{dataset_name}_results.png"))
  



def view_results(filename):
    results = load_results(filename)
    for key, value in results.items():
        print(f"{key}:")
        if isinstance(value, tuple):
            print(f"  NLPD: {value[0].mean():.4f} ± {value[0].std() / np.sqrt(10):.4f}")
            print(f"  RMSE: {value[1].mean():.4f} ± {value[1].std() / np.sqrt(10):.4f}")
        elif isinstance(value, list):
            for item in value:
                print(f"  num_projections: {item['num_projections']}")
                for proj_dim, nlpd, rmse, std_err_nlpd, std_err_rmse in zip(item['proj_dim_vals'], item['nlpd'], item['rmse'], item['std_err_nlpd'], item['std_err_rmse']):
                    print(f"    Proj. dim: {proj_dim}")
                    print(f"    NLPD: {nlpd:.4f} ± {std_err_nlpd:.4f}")
                    print(f"    RMSE: {rmse:.4f} ± {std_err_rmse:.4f}")
        print()


# # Configuration
# dataset_names = ["autos", 'housing', 'stock', 'breastcancer', 'forest', 'machine', 'yacht', 'autompg']
# num_proj_dim_vals = 6  # Number of different proj_dim values to test
# num_projections_vals = [2,5,10,20]  # Example values, you can modify

# # Run and store results
# run_and_store_all_datasets(dataset_names, num_proj_dim_vals, num_projections_vals)

# # Plot results
# # plot_results(dataset_names, num_proj_dim_vals)

# # View results
# # filename = "results/autos_results.pkl"  # Example filename
# # view_results(filename)
if __name__ == '__main__':
    save_path = "results_stack_rand_proj/"

    # Configuration
    dataset_names = ["autos", 'housing', 'stock', 'breastcancer', 'forest', 'machine', 'yacht', 'autompg']
    num_proj_dim_vals = 6  # Number of different proj_dim values to test
    num_projections_vals = [2,5,10,20]  # Example values, you can modify

    # Run and store results
    # run_and_store_all_datasets(dataset_names, num_proj_dim_vals, num_projections_vals,save_path)

    # Plot results
    # plot_results(dataset_names, num_proj_dim_vals,save_path=save_path)

    # View results
    dataset_name = dataset_names[1]
    filename = os.path.join(save_path, f"{dataset_name}_results.pkl")  # Example filename
    view_results(filename)