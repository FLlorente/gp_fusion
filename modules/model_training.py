import torch
import gpytorch
import numpy as np

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1], 
                                       lengthscale_prior=gpytorch.priors.GammaPrior(1, 1),
                                       ),
            outputscale_prior=gpytorch.priors.GammaPrior(1, 2)
            
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_and_predict_single_gp(X_train, y_train, X_test, X_val, kappa, lambdaa):
    torch.manual_seed(0)

    Xtrain_torch = torch.from_numpy(X_train).type(torch.float32)
    Ytrain_torch = torch.from_numpy(y_train).type(torch.float32).squeeze(-1)
    Xtest_torch = torch.from_numpy(X_test).type(torch.float32)
    Xval_torch = torch.from_numpy(X_val).type(torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
                # noise_prior=gpytorch.priors.GammaPrior(1, 1)
    )

    model_gpy = ExactGPModel(Xtrain_torch, Ytrain_torch, likelihood)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0 * y_train.var() / kappa**2),
        'covar_module.base_kernel.lengthscale': torch.from_numpy(np.std(X_train, axis=0) / lambdaa),
        'covar_module.outputscale': torch.tensor(1.0 * y_train.var()),
        'mean_module.constant': torch.tensor(y_train.mean(), requires_grad=False)
    }

    model_gpy.initialize(**hypers)

    training_iter = 100
    model_gpy.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model_gpy.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gpy)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model_gpy(Xtrain_torch)
        loss = -mll(output, Ytrain_torch)
        loss.backward()
        optimizer.step()

    model_gpy.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_preds = likelihood(model_gpy(Xtest_torch))
        val_preds = likelihood(model_gpy(Xval_torch))

    return test_preds, val_preds

def train_expert(X_train, y_train, kappa, lambdaa):
    torch.manual_seed(0)

    Xtrain_torch = torch.from_numpy(X_train).type(torch.float32)
    Ytrain_torch = torch.from_numpy(y_train).type(torch.float32).squeeze(-1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        # noise_prior=gpytorch.priors.GammaPrior(1, 1)
        )
    model_gpy = ExactGPModel(Xtrain_torch, Ytrain_torch, likelihood)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0 * y_train.var() / kappa ** 2),
        'covar_module.base_kernel.lengthscale': torch.from_numpy(np.std(X_train, axis=0) / lambdaa),
        'covar_module.outputscale': torch.tensor(1.0 * y_train.var()),
        'mean_module.constant': torch.tensor(y_train.mean(), requires_grad=False)
    }
    model_gpy.initialize(**hypers)

    training_iter = 100
    model_gpy.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model_gpy.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_gpy)

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model_gpy(Xtrain_torch)
        loss = -mll(output, Ytrain_torch)
        loss.backward()
        optimizer.step()

    model_gpy.eval()
    likelihood.eval()

    return model_gpy, likelihood

def predict_with_expert(model, likelihood, X):
    X_torch = torch.from_numpy(X).type(torch.float32)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_torch))
    return preds.mean.numpy(), np.sqrt(preds.variance.numpy())

def store_predictions_for_experts(experts, X):
    mu_preds = []
    std_preds = []
    for model, likelihood in experts:
        mu, std = predict_with_expert(model, likelihood, X)
        mu_preds.append(mu)
        std_preds.append(std)
    mu_preds = np.stack(mu_preds, axis=-1)
    std_preds = np.stack(std_preds, axis=-1)
    return mu_preds, std_preds
