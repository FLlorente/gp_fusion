# %%
import argparse
import numpy as np
import math
import torch
import gpytorch



# from uci_datasets import all_datasets
from uci_datasets import Dataset



parser = argparse.ArgumentParser(
    prog="single_gp",
)
# parser.add_argument("-M", "--num_models", default=2, type=int)
parser.add_argument("-i", "--i_split", default=0, type=int)
# parser.add_argument("-S", "--num_freq", default=50, type=int)
parser.add_argument("-D", "--dataset", default='yacht', type=str)

args = parser.parse_args()

print(f"Running single GP with split i={args.i_split} on {args.dataset}" )

i_split = args.i_split
# M_models = args.num_models
# S = args.num_freq


# %%
data = Dataset(args.dataset)
X_train, y_train, X_test, y_test = data.get_split(split=i_split)  # there are 10 different trainning-test splits

y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Normalize data to unit interval
X_train_max = X_train.max(0)
X_train_min = X_train.min(0)
X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

# standardize data to zero mean and unit variance
y_train_mean = y_train.mean(0)
y_train_std = y_train.std(0)

y_train = (y_train - y_train_mean)/y_train_std
y_test = (y_test - y_train_mean)/y_train_std

# %%
Xtrain_torch = torch.from_numpy(X_train)
Ytrain_torch = torch.from_numpy(y_train).squeeze(-1)

Xtest_torch = torch.from_numpy(X_test)
Ytest_torch = torch.from_numpy(y_test).squeeze(-1)

# %%
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
    'likelihood.noise_covar.noise': torch.tensor(0.25),
    'covar_module.base_kernel.lengthscale': torch.tensor(0.5)*torch.ones(X_train.shape[1],1),
    # 'covar_module.base_kernel.lengthscale': torch.tensor([1.6910,0.4038, 2.2482,2.2176,0.8974,0.2249]),
    'covar_module.outputscale': torch.tensor(1.),
}

model_gpy.initialize(**hypers);

# %%
training_iter = 100

# Find optimal model hyperparameters
model_gpy.train()
likelihood.train()   

# Use the adam optimizer
optimizer = torch.optim.Adam(model_gpy.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gpy)

print("starting learning of hyperparameters...")
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model_gpy(Xtrain_torch)  # this is for computing the prior GP model
    # Calc loss and backprop gradients
    loss = -mll(output, Ytrain_torch)
    loss.backward()
    optimizer.step()
print("learning finished!")

# %%
# Get into evaluation (predictive posterior) mode
model_gpy.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_preds = likelihood(model_gpy(Xtest_torch))    # prediccion de las ytest (no ftest) ya que estamos usando likelihood()
    train_preds = likelihood(model_gpy(Xtrain_torch)) # prediccion de las ytrain (no ftrain) ya que estamos usando likelihood()

# computing the average standardized negative log-likelihood of the points wrt their univariate predictive densities
test_lpd = -gpytorch.metrics.mean_standardized_log_loss(test_preds, Ytest_torch).numpy()
train_lpd = -gpytorch.metrics.mean_standardized_log_loss(train_preds, Ytrain_torch).numpy()

test_mse = gpytorch.metrics.mean_squared_error(test_preds, Ytest_torch, squared=True).numpy()
train_mse = gpytorch.metrics.mean_squared_error(train_preds, Ytrain_torch, squared=True).numpy()


ymu_single_gp = test_preds.mean.numpy()

# %%

np.savez(
    f"results_single_gp_{args.dataset}_i_{i_split}",
    single_gp_test_lpd = test_lpd,
    single_gp_train_lpd = train_lpd,
    single_gp_test_mse = test_mse,
    single_gp_train_mse = train_mse,
    y_test = y_test,
    ymu_single_gp = ymu_single_gp,
)
