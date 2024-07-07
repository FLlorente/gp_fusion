# This is the original self-contained script for running comparison between the stacking and
# product of expert algorithms. The functions in this script are outdated.


# %%
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, Predictive
# from numpyro.infer.autoguide import AutoDelta, AutoNormal
from optax import adam, chain, clip

import jax
import jax.numpy as jnp
import jax.random as random

import torch
import gpytorch

numpyro.set_host_device_count(4)
numpyro.set_platform("cpu")
numpyro.enable_x64()


# %% UCI regression datasets 

from uci_datasets import all_datasets
# all_datasets is a dictionary that contains the names of all datasets and info about the number of observations and dimension of the input vector
# print(all_datasets.keys())

# %% Loading the data and getting the train-test split
from uci_datasets import Dataset
data = Dataset('yacht')

X_train, y_train, X_test, y_test = data.get_split(split=5)  # there are 10 different trainning-test splits
y_train = y_train.squeeze(); y_test = y_test.squeeze()

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Normalize data to unit interval
X_train_max = X_train.max(0)
X_train_min = X_train.min(0)
X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)


# %% This function is for fitting a GP and making predictions; kappa and lambdaa control the initial hyperparameters (noise and lengthscale)
def train_and_predict_single_gp(X_train,y_train,X_test,X_val,kappa,lambdaa):
    Xtrain_torch = torch.from_numpy(X_train).type(torch.float32)
    Ytrain_torch = torch.from_numpy(y_train).type(torch.float32).squeeze(-1)
    Xtest_torch = torch.from_numpy(X_test).type(torch.float32)
    Xval_torch = torch.from_numpy(X_val).type(torch.float32)

    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            lengthscale_prior = gpytorch.priors.GammaPrior(1, 1)
            outputscale_prior = gpytorch.priors.GammaPrior(1, 2)

            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=Xtrain_torch.shape[1],
                lengthscale_prior = lengthscale_prior
                )
                ,outputscale_prior = outputscale_prior,
                )
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=gpytorch.priors.GammaPrior(1, 1))
    model_gpy = ExactGPModel(Xtrain_torch, Ytrain_torch, likelihood)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0*y_train.var()/kappa**2),
        'covar_module.base_kernel.lengthscale': torch.from_numpy(np.std(X_train,axis=0)/lambdaa),
        'covar_module.outputscale': torch.tensor(1.0*y_train.var()),
        'mean_module.constant': torch.tensor(y_train.mean())
    }
    model_gpy.initialize(**hypers);

    training_iter = 100

    model_gpy.train()
    likelihood.train()   

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model_gpy.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs: - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gpy)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model_gpy(Xtrain_torch)  # this is for computing the prior GP model
        # Calc loss and backprop gradients
        loss = -mll(output, Ytrain_torch)
        loss.backward()
        optimizer.step()

    # Get into evaluation (predictive posterior) mode
    model_gpy.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_preds = likelihood(model_gpy(Xtest_torch))    
        train_preds = likelihood(model_gpy(Xtrain_torch))
        val_preds = likelihood(model_gpy(Xval_torch))    


    return test_preds, train_preds, val_preds
   

# %% This function is for splitting the training dataset into subsets that will be used for training the experts
def split_dataset(X, Y, n_splits, split_size, with_replacement=True):
    np.random.seed(0)
    n_samples = X.shape[0]

    if with_replacement:
        splits = [(X[indices], Y[indices]) for indices in [np.random.choice(n_samples, split_size, replace=True) for _ in range(n_splits)]]
    else:
        if split_size * n_splits > n_samples:
            raise ValueError("Cannot split without replacement as there are not enough samples.")
        indices = np.random.permutation(n_samples)
        splits = [(X[indices[i * split_size:(i + 1) * split_size]], Y[indices[i * split_size:(i + 1) * split_size]]) for i in range(n_splits)]
    
    return splits



# %% This function is for creating the validation dataset, which is made of some observations from each of the experts' training data

def create_validation_set(splits, n_points_per_split):
    np.random.seed(0)
    X_val = []
    y_val = []
    
    # Iterate over each split
    for X_split, y_split in splits:
        
        # Randomly select n_points_per_split indices from the split
        selected_indices = np.random.choice(len(X_split), n_points_per_split, replace=False)
        
        # Add the selected points to the validation set
        X_val.append(X_split[selected_indices])
        y_val.append(y_split[selected_indices])
        
    # Combine the validation points from all splits
    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    
    return X_val, y_val

n_experts = 4
n_data_per_expert = X_train.shape[0]//n_experts

splits = split_dataset(X_train, y_train, 
                       n_splits=n_experts, 
                       split_size=n_data_per_expert, 
                      #  split_size = 100,
                      #  with_replacement=True,  
                      with_replacement=False,    # partition
                       )

n_points_per_split = 5  # number of datapoints that we take from each of the experts' split
X_val, y_val = create_validation_set(splits, n_points_per_split)

# %% Here we fit the n_experts GPs and store the predictions on test and validation data

mu_preds_val = np.zeros((X_val.shape[0],len(splits)))
std_preds_val = np.zeros((X_val.shape[0],len(splits)))

mu_preds_test = np.zeros((X_test.shape[0],len(splits)))
std_preds_test = np.zeros((X_test.shape[0],len(splits)))

kappa = 2  # in [2,100]
lambdaa = 1 # in [1,10]
# cmap = matplotlib.colormaps['viridis']
for i, (X_split, y_split) in enumerate(splits):
   test_preds, _, val_preds = train_and_predict_single_gp(X_train=X_split,
                                y_train=y_split,
                                X_test=X_test,
                                X_val = X_val,
                                kappa=kappa,
                                lambdaa = lambdaa)
   
   mu_preds_val[:,i] = val_preds.mean.numpy()
   std_preds_val[:,i] = np.sqrt(val_preds.variance.numpy())

   mu_preds_test[:,i] = test_preds.mean.numpy()
   std_preds_test[:,i] = np.sqrt(test_preds.variance.numpy())



# %%
# compute negative test loglikelihood for every expert before the fusion

def compute_neg_log_like(mus,stds,y_test):
    negloglik = np.zeros((y_test.shape[0],mus.shape[1]))
    for i in range(mus.shape[1]):
       negloglik[:,i] = -1.0*scipy.stats.norm.logpdf(y_test, mus[:,i],stds[:,i])

    return negloglik.mean(0)   

nlpd_experts = compute_neg_log_like(mu_preds_test,std_preds_test,y_test)


# %% This is single GP using all training data for comparison purposes
test_preds, _, _ = train_and_predict_single_gp(
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                X_val = X_val,
                                kappa=kappa,
                                lambdaa=lambdaa)

nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1,1),
                                   np.sqrt(test_preds.variance.numpy().reshape(-1,1)),
                                   y_test)



# %%
# fusion of GP predictions using log-linear pooling with weights computed "at prediction time" (a la gPoEs, rBCM, etc.)

def product_fusion(mus,stds,X_test):
    prec_fused = np.zeros((X_test.shape[0],1))
    mean_fused = np.zeros((X_test.shape[0],1))
    w_gpoe = np.zeros(mus.shape)
    #
    noise = np.var(y_train)/kappa**2
    variance = np.var(y_train)
    for n, x in enumerate(X_test):

        weights = 0.5*(np.log(noise+variance) - np.log(stds[n,:]**2))
        weights = weights / np.sum(weights)

        precs = 1/stds[n,:]**2

        prec_fused[n,:] = weights @ precs
        mean_fused[n,:] = weights @ (mus[n,:]*precs) / prec_fused[n,:]

        # store weights
        w_gpoe[n,:] = weights

    return mean_fused, 1/np.sqrt(prec_fused), w_gpoe

# %%
mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test,std_preds_test,X_test)

# %%
nlpd_gpoe = compute_neg_log_like(mus_gpoe,
                                 stds_gpoe,
                                 y_test)
print("nlpd gpoe: ", nlpd_gpoe)

# %%
plt.figure()
for i in range(w_gpoe.shape[1]):
    plt.plot(w_gpoe[:,i])

# %% PHS: this method implements log-linear fusion with learned weights

# squared euclidean distance
def sqeuclidean_distance(x, y):
    return jnp.sum((x - y) ** 2)

# distance matrix
def cross_covariance(func, x, y):
    """distance matrix"""
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)

def SE_kernel(X, Y, var, length, noise, jitter=1.0e-6, include_noise=True):
    # distance formula
    deltaXsq = cross_covariance(
        sqeuclidean_distance, X / length, Y / length
    )

    assert deltaXsq.shape == (X.shape[0], Y.shape[0])

    # rbf function
    K = var * jnp.exp(-0.5 * deltaXsq)
    if include_noise:
        K += (noise + jitter) * jnp.eye(X.shape[0])
    return K

vmap_SE_kernel = jax.vmap(SE_kernel, in_axes=(None, None, 0, 0, 0))

def predict_with_mean(
    rng_key,
    X,
    Y,
    X_test,
    var,
    length,
    noise,
    kernel_func=SE_kernel,
    mean_func=lambda x: jnp.zeros(x.shape[0]),
):
    # compute kernels between train and test data, etc.
    k_pp = kernel_func(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = kernel_func(X_test, X, var, length, noise, include_noise=False)
    k_XX = kernel_func(X, X, var, length, noise, include_noise=True)
    K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
    K = k_pp - jnp.matmul(
        k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, jnp.transpose(k_pX))
    )
    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), a_min=0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )
    mean = mean_func(X_test) + jnp.matmul(
        k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y - mean_func(X))
    )
    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, jnp.sqrt(jnp.diag(K))


vmapped_pred_with_mean = jax.vmap(
    predict_with_mean, in_axes=(None, None, 0, None, 0, 0, 0, None, None)
)

vmapped_pred_with_mean = jax.vmap(
    vmapped_pred_with_mean,
    in_axes=(None, None, 1, None, 1, 1, 1, None, None),
)

def phs(X, mu_preds, std_preds, y_val=None):
    N, M = mu_preds.shape

    assert mu_preds.shape == std_preds.shape

    tau_preds = 1 / std_preds**2

    ######################
    # GP for log weights #
    ######################
    with numpyro.plate("M", M):
        kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(1.0))
        kernel_length = numpyro.sample("kernel_length", dist.InverseGamma(5.0, 5.0))
        kernel_noise = numpyro.sample("kernel_noise", dist.HalfNormal(1.0))

    k = numpyro.deterministic(
        "k", vmap_SE_kernel(X, X, kernel_var, kernel_length, kernel_noise)
    )

    with numpyro.plate("logw_plate", M, dim=-1):
        log_w = numpyro.sample(
            "w_un", dist.MultivariateNormal(loc=-jnp.log(M), covariance_matrix=k)
        )

    ################################################
    # Fuse with generalized multiplicative pooling #
    ################################################
    w = numpyro.deterministic("w", jnp.exp(log_w))

    tau_fused = numpyro.deterministic(
        "tau_fused", jnp.einsum("nm,mn->n", tau_preds, w)
    )  # N,

    assert tau_fused.shape == (N,)
    mu_fused = numpyro.deterministic(
        "mean_fused", jnp.einsum("nm,nm,mn->n", tau_preds, mu_preds, w) / tau_fused
    )  # N,
    assert mu_fused.shape == (N,)
    std_fused = numpyro.deterministic("std_fused", 1 / jnp.sqrt(tau_fused))

    numpyro.sample(
        "y_val",
        dist.Normal(loc=jnp.squeeze(mu_fused), scale=jnp.squeeze(std_fused)),
        obs=y_val,
    )

    numpyro.deterministic(
        "lpd_point",
        jax.scipy.stats.norm.logpdf(
        jnp.squeeze(y_val), loc=jnp.squeeze(mu_fused), scale=jnp.squeeze(std_fused))    
    )

# %% These are functions for carrying out the training and prediction of the fusion methods that need learning of the weights
def train_stacking(model=None):
    mcmc = MCMC(
    NUTS(model, 
         init_strategy=numpyro.infer.initialization.init_to_median,
         ),
    num_warmup=100,
    num_samples=100,
    num_chains=4,
)

    mcmc.run(random.PRNGKey(0), 
            X_val,   
            mu_preds_val, 
            std_preds_val, 
            y_val=y_val, 
            )
    mcmc.print_summary()
    samples = mcmc.get_samples()

    return samples


def predict_stacking(model,samples,prior_mean = lambda x:  jnp.zeros(x.shape[0])):

    res = vmapped_pred_with_mean(
        random.PRNGKey(0),
        X_val,
        samples["w_un"],
        X_test, # TEST DATA
        samples["kernel_var"],
        samples["kernel_length"],
        samples["kernel_noise"],
        SE_kernel,
        prior_mean,
    )


    # w_un_samples = jnp.asarray(res[0] + np.random.randn(*res[0].shape) * res[1])
    w_un_samples = jnp.asarray(res[0])
    pred_samples = {"w_un": jnp.transpose(w_un_samples, (1, 0, 2))}

    predictive = Predictive(model, pred_samples)
    pred_samples = predictive(
        random.PRNGKey(0),
        X=X_test, # TEST DATA
        mu_preds=mu_preds_test,  # TEST DATA
        std_preds=std_preds_test, # TEST DATA
        y_val = y_test, # TEST DATA
    )

    lpd_test = jax.nn.logsumexp(pred_samples["lpd_point"],axis=0) - np.log(pred_samples["lpd_point"].shape[0])

    return pred_samples, lpd_test


# %%
samples_phs = train_stacking(
    model=phs,
    )
preds_phs, lpd_phs_test = predict_stacking(
                                           model = phs,
                                           samples=samples_phs,
                                           prior_mean=lambda x: -jnp.log(mu_preds_test.shape[1]) * jnp.ones(x.shape[0]),
                                           )

# %%
print("nlpd PHS: ", -lpd_phs_test.mean())

plt.figure()
for i in range(n_experts):
    plt.plot(preds_phs["w"].mean(0)[i,:])

# %% BHS: this is another fusion method where we learn the weightsl; differently from PHS, it uses linear pooling

def bhs(X, mu_preds, std_preds, y_val=None):
    N, M = mu_preds.shape

    assert mu_preds.shape == std_preds.shape

    ######################
    # GP for log weights #
    ######################
    with numpyro.plate("M", M):
        kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(1.0))
        kernel_length = numpyro.sample(
            "kernel_length", dist.InverseGamma(5.0, 5.0))
        kernel_noise = numpyro.sample("kernel_noise", dist.HalfNormal(1.0))

    k = numpyro.deterministic(
        "k", vmap_SE_kernel(X, X, kernel_var, kernel_length, kernel_noise)
    )

    with numpyro.plate("logw_plate", M, dim=-1):
        w_un = numpyro.sample(
            "w_un", dist.MultivariateNormal(
                loc=-jnp.log(M), covariance_matrix=k)
        )

    log_w = jax.nn.log_softmax(w_un.T, axis=1)

    #################
    # Fuse with BHS #
    #################
    y_val_rep = jnp.tile(jnp.reshape(y_val, (-1, 1)), M)
    lpd_point = jax.scipy.stats.norm.logpdf(
        y_val_rep, loc=mu_preds, scale=std_preds)
    logp = jax.nn.logsumexp(lpd_point + log_w, axis=1)
    numpyro.deterministic("lpd_point", logp)
    numpyro.deterministic("w", jnp.exp(log_w))
    numpyro.factor("logp", jnp.sum(logp))


# %%
samples_bhs = train_stacking(model=bhs)
preds_bhs, lpd_bhs_test = predict_stacking(bhs,samples=samples_bhs)

# %%
print("nlpd BHS: ", -lpd_bhs_test.mean())

plt.figure()
for i in range(n_experts):
    plt.plot(preds_bhs["w"].mean(0)[:,i])




