import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


# sys.path.append('/Users/fllorente/Dropbox/con_Petar/PYTHON/gp_fusion')

from modules.data_handling import load_data, load_and_normalize_data
from modules.model_training import train_and_predict_single_gp
from modules.model_training import GPModel, to_torch
from modules.fusion_methods import product_fusion
from modules.fusion_methods import compute_neg_log_like
from modules.model_training import train_expert, predict_with_expert, train_and_predict_batched_gp


import torch
from tqdm import tqdm

from gpytorch.means import ZeroMean
from gpytorch.kernels import AdditiveStructureKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.metrics import mean_standardized_log_loss


import numpyro
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from optax import adam, chain, clip

from numpyro import distributions as dist
import jax
import jax.numpy as jnp
import jax.random as random

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)
numpyro.enable_x64()

from uci_datasets import Dataset



def run_single_gp(dataset_name, kernel = None, lr = 0.1, training_iter=100):
    nlpd = []
    rmse = []
    for i in tqdm(range(10)):
        try:
            X_train, y_train, X_test, y_test, y_std = load_data(dataset_name,i)

            # With load_and_normalize_data fun the data is normalized using the training data only
            # X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name,i,
            #                                                         normalize_y=True,
            #                                                         normalize_x_method="z-score")

            test_preds, _ = train_and_predict_single_gp(X_train,y_train,X_test,X_test,
                                    mean=ZeroMean(),  # we don't want to use the mean of y_train as prior mean
                                    kernel = kernel,
                                    lr=lr,
                                    training_iter=training_iter,
                                    initialiaze_hyper=False, # if false, kappa and lambdaa don't matter!
                                    )
            nlpd_now = compute_neg_log_like(test_preds.mean,np.sqrt(test_preds.variance),y_test)
            rmse_now = np.sqrt(np.mean((test_preds.mean.numpy().squeeze() - y_test)**2))

            nlpd.append(nlpd_now.squeeze())
            rmse.append(rmse_now)
        except:
            print("There was an error during hyperparameter learning.")

    nlpd = np.array(nlpd)
    rmse = np.array(rmse)

    return nlpd, rmse


def run_gam_gp(dataset_name, lr = 0.1, training_iter = 100):

    nlpd = []
    rmse = []
    for i in tqdm(range(10)):
        try:
            X_train, y_train, X_test, y_test, y_std = load_data(dataset_name,i)

            # With load_and_normalize_data fun the data is normalized using the training data only
            # X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name,i,
            #                                                         normalize_y=True,
            #                                                         normalize_x_method="z-score")
              

            kernel = AdditiveStructureKernel(base_kernel=ScaleKernel(RBFKernel()), 
                                             num_dims=X_train.shape[1])

            test_preds, _ = train_and_predict_single_gp(X_train,y_train,X_test,X_test,
                                    kappa=2,lambdaa=2,
                                    mean=ZeroMean(),  # we don't want to use the mean of y_train as prior mean
                                    kernel=kernel,
                                    lr=lr,
                                    training_iter=training_iter,
                                    initialiaze_hyper=False, # if False, kappa and lambdaa are not used for initializing the hyperparameters; we just use the default values.
                                    )
            nlpd_now = compute_neg_log_like(test_preds.mean,np.sqrt(test_preds.variance),y_test)
            rmse_now = np.sqrt(np.mean((test_preds.mean.detach().numpy().squeeze() - y_test)**2))

            nlpd.append(nlpd_now.squeeze())
            rmse.append(rmse_now)
        except:
            print("There was an error during hyperparameter learning.")

    nlpd = np.array(nlpd)
    rmse = np.array(rmse)

    return nlpd, rmse



def run_stacked_proj_gp(dataset_name, num_projections, project_dim,
                        proj = "random",lr = 0.1, training_iter = 100,
                        ):
    results = defaultdict(lambda: {"nlpd": [], "rmse": []})

    configs = [
        {'weighting': 'uniform', 'method': 'gPoE', 'normalize': True},
        {'weighting': 'entropy', 'method': 'gPoE', 'normalize': True},
        {'weighting': 'entropy', 'method': 'gPoE', 'normalize': False},
        {'weighting': 'entropy', 'method': 'rBCM', 'normalize': False}
    ]

    for split in tqdm(range(10)):
        X_train, y_train, X_test, y_test, y_std = load_data(dataset_name, split)

        # With load_and_normalize_data fun the data is normalized using the training data only
        # X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name,split,
        #                                                             normalize_y=True,
        #                                                             normalize_x_method="z-score")

        mean_experts = []
        std_experts = []
        std_experts_prior = []

        for d in range(num_projections):  
            if proj == "random":
                np.random.seed(d)
                P_proj = np.random.randn(X_train.shape[1], project_dim) / np.sqrt(project_dim)
            elif proj == "axis":
                P_proj = np.array([1.0 if i == d else 0.0 for i in range(X_train.shape[1])])
                P_proj = P_proj.reshape(-1, 1)

            model, likelihood = train_expert(np.matmul(X_train, P_proj),
                                             y_train,
                                             mean=ZeroMean(),
                                             lr=lr,
                                             training_iter=training_iter,
                                             initialize_hyper=False)

            mean_preds, std_preds, std_preds_prior = predict_with_expert(model,
                                                                         likelihood,
                                                                         np.matmul(X_test, P_proj))

            mean_experts.append(mean_preds)
            std_experts.append(std_preds)
            std_experts_prior.append(std_preds_prior)

        mean_experts = np.stack(mean_experts, axis=-1)
        std_experts = np.stack(std_experts, axis=-1)
        std_experts_prior = np.stack(std_experts_prior, axis=-1)

        for config in configs:
            mean_fused, std_fused, _ = product_fusion(mean_experts,
                                                      std_experts,
                                                      std_experts_prior,
                                                      weighting=config["weighting"],
                                                      normalize=config["normalize"],
                                                      method=config["method"])

            nlpd_now = compute_neg_log_like(mean_fused, std_fused, y_test).squeeze()
            rmse_now = np.sqrt(np.mean((mean_fused.squeeze() - y_test.squeeze())**2))

            # Store the results
            config_key = f'{config["method"]}_{config["weighting"]}_normalize_{config["normalize"]}'
            results[config_key]["nlpd"].append(nlpd_now)
            results[config_key]["rmse"].append(rmse_now)

    # Calculate the mean performance measures across all splits for each configuration
    mean_results = {key: {"mean_nlpd": np.mean(value["nlpd"]), 
                          "mean_rmse": np.mean(value["rmse"]),
                          "std_err_nlpd": np.std(value["nlpd"])/np.sqrt(10), 
                          "std_err_rmse": np.std(value["rmse"])/np.sqrt(10),
                          }
                    for key, value in results.items()}

    return mean_results







def run_stacked_proj_gp_batched(dataset_name, num_projections, project_dim,
                        proj = "random",lr = 0.1, training_iter = 100,
                        ):
    results = defaultdict(lambda: {"nlpd": [], "rmse": []})

    configs = [
        {'weighting': 'uniform', 'method': 'gPoE', 'normalize': True},
        {'weighting': 'entropy', 'method': 'gPoE', 'normalize': True},
        {'weighting': 'entropy', 'method': 'gPoE', 'normalize': False},
        {'weighting': 'entropy', 'method': 'rBCM', 'normalize': False}
    ]

    for split in tqdm(range(10)):
        X_train, y_train, X_test, y_test, y_std = load_data(dataset_name, split)

        # With load_and_normalize_data fun the data is normalized using the training data only
        # X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name,split,
        #                                                             normalize_y=True,
        #                                                             normalize_x_method="z-score")

        mean_experts = []
        std_experts = []
        std_experts_prior = []

        X_train_batched = np.zeros((num_projections,X_train.shape[0],project_dim))
        X_test_batched = np.zeros((num_projections,X_test.shape[0],project_dim))
        y_train_batched = np.zeros((num_projections,y_train.shape[0]))
        for d in range(num_projections):  
            if proj == "random":
                np.random.seed(d)
                P_proj = np.random.randn(X_train.shape[1], project_dim) / np.sqrt(project_dim)
            elif proj == "axis":
                P_proj = np.array([1.0 if i == d else 0.0 for i in range(X_train.shape[1])])
                P_proj = P_proj.reshape(-1, 1)

            X_train_batched[d,:,:] = np.matmul(X_train,P_proj)
            y_train_batched[d,:] = y_train

            X_test_batched[d,:,:] = np.matmul(X_test,P_proj)


        # train all the experts in batch mode
        preds, preds_prior = train_and_predict_batched_gp(X_train_batched, 
                                                          y_train_batched, 
                                                          X_test_batched,
                                                          training_iter=training_iter, lr=lr)


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
            rmse_now = np.sqrt(np.mean((mean_fused.squeeze() - y_test.squeeze())**2))

            # Store the results
            config_key = f'{config["method"]}_{config["weighting"]}_normalize_{config["normalize"]}'
            results[config_key]["nlpd"].append(nlpd_now)
            results[config_key]["rmse"].append(rmse_now)

    # Calculate the mean performance measures across all splits for each configuration
    mean_results = {key: {"mean_nlpd": np.mean(value["nlpd"]), 
                          "mean_rmse": np.mean(value["rmse"]),
                          "std_err_nlpd": np.std(value["nlpd"])/np.sqrt(10), 
                          "std_err_rmse": np.std(value["rmse"])/np.sqrt(10),
                          }
                    for key, value in results.items()}

    return mean_results






dataset_names = ["autos", 'housing','stock','sml',
                 'elevators','breastcancer','forest','gas',
                 ]


dataset_name = "autos"


data = Dataset(dataset_name)
N,DIM = data.x.shape


nlpd1, rmse1  = run_single_gp(dataset_name)
nlpd2, rmse2  = run_gam_gp(dataset_name)


num_projections = 5
project_dim = 10
# results = run_stacked_proj_gp(dataset_name, num_projections, project_dim)

results_batched = run_stacked_proj_gp_batched(dataset_name, num_projections, project_dim)


# single GP results
print(f"{'single gp':<15} nlpd: {nlpd1.mean():.2f} ± {nlpd1.std()/np.sqrt(10):.2f}, rmse: {rmse1.mean():.2f} ± {rmse1.std()/np.sqrt(10):.2f}")

# gam GP results
print(f"{'gam gp':<15} nlpd: {nlpd2.mean():.2f} ± {nlpd2.std()/np.sqrt(10):.2f}, rmse: {rmse2.mean():.2f} ± {rmse2.std()/np.sqrt(10):.2f}")

# # results stacked randomly-projected experts
# for key in results.keys():
#     print(f"{key:<15} nlpd: {results[key]['mean_nlpd']:.2f} ± {results[key]['std_err_nlpd']:.2f}, rmse: {results[key]['mean_rmse']:.2f} ± {results[key]['std_err_rmse']:.2f}")

for key in results_batched.keys():
    print(f"{key:<15} nlpd: {results_batched[key]['mean_nlpd']:.2f} ± {results_batched[key]['std_err_nlpd']:.2f}, rmse: {results_batched[key]['mean_rmse']:.2f} ± {results_batched[key]['std_err_rmse']:.2f}")






