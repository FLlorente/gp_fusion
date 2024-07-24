import gpytorch.constraints
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np



def to_torch(x, dtype=torch.float32):
    return torch.from_numpy(x).type(dtype)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean=None):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = kernel or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def initialize_hyperparameters(model, likelihood, X_train, y_train, kappa, lambdaa):
    # Initialize likelihood noise parameter
    likelihood.noise_covar.initialize(noise=torch.tensor(1.0 * y_train.var() / kappa**2))
    
    # Initialize model kernel and mean parameters
    if isinstance(model.mean_module,gpytorch.means.ConstantMean):
        hypers = {
            'covar_module.base_kernel.lengthscale': torch.from_numpy(np.std(X_train, axis=0) / lambdaa),
            'covar_module.outputscale': torch.tensor(1.0 * y_train.var()),
            'mean_module.constant': torch.tensor(y_train.mean(), requires_grad=False)
        }
    elif isinstance(model.mean_module,gpytorch.means.ZeroMean):
        hypers = {
            'covar_module.base_kernel.lengthscale': torch.from_numpy(np.std(X_train, axis=0) / lambdaa),
            'covar_module.outputscale': torch.tensor(1.0 * y_train.var()),
        }
    else:
        raise ValueError("Mean functions is not of type ConstantMean or ZeroMean.")
    
    model.initialize(**hypers)

def train_model(model, likelihood, X_train, y_train, training_iter, lr, seed):
    torch.manual_seed(seed) # no parece afectar a los resultados...
    model.train()
    likelihood.train()
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()

def predict_with_expert(model, likelihood, X):
    X_torch = to_torch(X)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_torch))
        preds_prior = likelihood(model.forward(X_torch))
    return preds.mean.numpy(), np.sqrt(preds.variance.numpy()), np.sqrt(preds_prior.variance.numpy())

def store_predictions_for_experts(experts, X):
    mu_preds, std_preds, std_preds_prior = [], [], []
    for model, likelihood in experts:
        mu, std, std_prior = predict_with_expert(model, likelihood, X)
        mu_preds.append(mu)
        std_preds.append(std)
        std_preds_prior.append(std_prior)
    return np.stack(mu_preds, axis=-1), np.stack(std_preds, axis=-1), np.stack(std_preds_prior, axis=-1)

def train_and_predict_single_gp(X_train, y_train, X_test, X_val, kappa=2.0, lambdaa=1.0, kernel=None, mean=None, training_iter=100, lr=0.1, seed = 0, initialiaze_hyper = True):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )

    model_gpy = GPModel(to_torch(X_train), to_torch(y_train).squeeze(-1), likelihood, kernel, mean)
    
    if initialiaze_hyper:
        initialize_hyperparameters(model_gpy, likelihood, X_train, y_train, kappa, lambdaa)
    
    train_model(model_gpy, likelihood, to_torch(X_train), to_torch(y_train).squeeze(-1), training_iter, lr, seed)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_preds = likelihood(model_gpy(to_torch(X_test)))
        val_preds = likelihood(model_gpy(to_torch(X_val)))

    return test_preds, val_preds

def train_expert(X_train, y_train, kappa=2.0, lambdaa=1.0, kernel=None, mean=None, training_iter=100, lr=0.1, seed=0,initialize_hyper=True):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model_gpy = GPModel(to_torch(X_train), to_torch(y_train).squeeze(-1), likelihood, kernel, mean)
    
    if initialize_hyper:
        initialize_hyperparameters(model_gpy, likelihood, X_train, y_train, kappa, lambdaa)
    
    
    train_model(model_gpy, likelihood, to_torch(X_train), to_torch(y_train).squeeze(-1), training_iter, lr, seed)

    return model_gpy, likelihood

def train_joint_experts_shared_kernel(expert_datasets, kappa=2.0, lambdaa=1.0, kernel=None, mean=None, training_iter=100, lr=0.1,seed=0):
    torch.manual_seed(seed)
    kernel = kernel or gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=expert_datasets[0][0].shape[1]))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    models = [GPModel(to_torch(X), to_torch(y).squeeze(-1), likelihood, kernel, mean) for X, y in expert_datasets]
    X_train = np.concatenate([X for X, y in expert_datasets], axis=0)
    y_train = np.concatenate([y for X, y in expert_datasets], axis=0)
    
    initialize_hyperparameters(models[0], likelihood, X_train, y_train, kappa, lambdaa)  # Initialize shared kernel and likelihood

    # for model in models:
    #     model.mean_module.initialize(constant=y_train.mean())

    optimizer = torch.optim.Adam([
        {'params': kernel.parameters()},  # Shared kernel parameters
        {'params': likelihood.parameters()},  # Likelihood parameters
    ], lr=lr)

    for _ in range(training_iter):
        optimizer.zero_grad()
        total_loss = 0
        for model in models:  # in principle, this loop could be distributed and/or parallelized
            train_x = model.train_inputs[0]
            train_y = model.train_targets
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            output = model(train_x)
            loss = -mll(output, train_y)
            total_loss += loss
        total_loss.backward()
        optimizer.step()
        # optimizer.zero_grad()

    for model in models:
        model.eval()
    likelihood.eval()

    return models, likelihood



class VariationalGPModel(ApproximateGP):
    def __init__(self, train_x, inducing_points, likelihood, kernel=None, mean=None):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = mean or ConstantMean()
        self.covar_module = kernel or ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def train_variational_gp(X_train, y_train, inducing_points, kappa=2.0, lambdaa=1.0, learning_rate=0.01, num_epochs=10, batch_size=128,seed=0,kernel=None, mean=None):
    torch.manual_seed(seed)
    likelihood = GaussianLikelihood()
    model = VariationalGPModel(to_torch(X_train), to_torch(inducing_points), likelihood,kernel=kernel,mean=mean)

    initialize_hyperparameters(model, likelihood, X_train, y_train, kappa, lambdaa)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = VariationalELBO(likelihood, model, num_data=X_train.shape[0])
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(to_torch(X_train), to_torch(y_train).squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    '''
    #Calculate the total number of iterations that will be run
    N = len(X_train)  # Total number of training samples
    B = batch_size    # Batch size
    num_batches = np.ceil(N / B)
    print("total number of iterations (gradient updates) will be ",num_batches * num_epochs)
    '''

    for i in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
    
    return model, likelihood  


def predict_variational_gp(model, likelihood, X_test, batch_size=128):
    model.eval()
    likelihood.eval()

    test_dataset = TensorDataset(to_torch(X_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    means = []
    variances = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for X_batch in test_loader:
            X_batch = X_batch[0]  # Unpack the tuple
            preds = likelihood(model(X_batch))
            means.append(preds.mean.cpu().numpy())
            variances.append(preds.variance.cpu().numpy())
    
    mean = np.concatenate(means, axis=0)
    var = np.concatenate(variances, axis=0)
    
    return mean, np.sqrt(var)



class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([train_x.shape[0]]))
        self.covar_module = kernel or gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_x.shape[0]]),
                                       ard_num_dims=train_x.shape[2]),
            batch_shape=torch.Size([train_x.shape[0]])
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


def train_and_predict_batched_gp(X_train, y_train, X_test,training_iter=100, lr=0.1, kernel = None):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
    
    X_train = to_torch(X_train)
    y_train = to_torch(y_train)
    X_test = to_torch(X_test)


    assert X_train.ndim == 3
    assert y_train.ndim == 2
    assert X_test.ndim == 3

    model = BatchGPModel(X_train, y_train, likelihood, kernel)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output,y_train).sum()
        loss.backward()
        optimizer.step()


    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test))
        preds_prior = likelihood(model.forward(X_test))

    return preds, preds_prior




class BatchVariationalGPModel(ApproximateGP):
    def __init__(self, train_x,num_inducing_points, kernel=None, mean=None):
        inducing_points = train_x[:,:num_inducing_points,:]
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points,
                                                                   batch_shape=torch.Size([train_x.shape[0]]))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(BatchVariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = mean or ZeroMean(batch_shape=torch.Size([train_x.shape[0]]))
        self.covar_module = kernel or ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[2],
                                                            batch_shape=torch.Size([train_x.shape[0]])),
                                                  batch_shape=torch.Size([train_x.shape[0]]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def train_and_predict_batched_svgp(X_train, y_train, X_test,num_epochs=5, lr=0.1, batch_size = 128, num_inducing_points = 100):
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        batch_shape=torch.Size([X_train.shape[0]]),
        )
    
    X_train = to_torch(X_train)
    y_train = to_torch(y_train)
    X_test = to_torch(X_test)

    assert X_train.ndim == 3
    assert y_train.ndim == 2
    assert X_test.ndim == 3

    model = BatchVariationalGPModel(X_train,num_inducing_points=num_inducing_points)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    mll = VariationalELBO(likelihood, model, num_data=X_train.shape[1])
    
    # Create DataLoader for mini-batch training
    train_dataset = TensorDataset(X_train.transpose(0, 1), y_train.transpose(0, 1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    '''
    #Calculate the total number of iterations that will be run
    N = len(X_train)  # Total number of training samples
    B = batch_size    # Batch size
    num_batches = np.ceil(N / B)
    print("total number of iterations (gradient updates) will be ",num_batches * num_epochs)
    '''

    for i in range(num_epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.transpose(0, 1)
            y_batch = y_batch.transpose(0, 1)

            # print(X_batch.shape)
            # print(y_batch.shape)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch).sum()
            loss.backward()
            optimizer.step()


    model.eval()
    likelihood.eval()

    test_dataset = TensorDataset(X_test.transpose(0, 1))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    means = []
    variances = []
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for X_batch in test_loader:
            X_batch = X_batch[0]  # Unpack the tuple
            X_batch = X_batch.transpose(0, 1)
            
            preds = likelihood(model(X_batch))
            means.append(preds.mean.cpu().numpy())
            variances.append(preds.variance.cpu().numpy())
    
    mean = np.concatenate(means, axis=1)
    var = np.concatenate(variances, axis=1)
    
    return mean, np.sqrt(var)


    