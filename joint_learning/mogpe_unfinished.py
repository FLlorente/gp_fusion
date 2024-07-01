# %%
import argparse
import numpyro.infer.initialization
from tqdm import tqdm

import numpy as np

import numpyro
from numpyro.infer import NUTS, MCMC, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta, AutoNormal

from optax import adam, chain, clip

from numpyro import distributions as dist
import jax
import jax.numpy as jnp
import jax.random as random

# from uci_datasets import all_datasets
from uci_datasets import Dataset

numpyro.set_host_device_count(4)
numpyro.set_platform("cpu")
numpyro.enable_x64()

parser = argparse.ArgumentParser(
    prog="pogpe",
)
parser.add_argument("-M", "--num_models", default=2, type=int)
parser.add_argument("-i", "--i_split", default=0, type=int)
parser.add_argument("-S", "--num_freq", default=50, type=int)
parser.add_argument("-D", "--dataset", default='yacht', type=str)

args = parser.parse_args()

print(f"Running with S={args.num_freq} i={args.i_split} M={args.num_models} on {args.dataset}" )

i_split = args.i_split
M_models = args.num_models
S = args.num_freq


def myfun(A, B):
    return (A @ B).squeeze()


matmul_vmapped = jax.vmap(myfun, in_axes=(0, 0))

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

DIM = X_train.shape[1]  # dimension of the input vector
Omega_fixed = jax.random.normal(jax.random.PRNGKey(i_split + 100), (S, DIM))


class RandomHandler:
    def __init__(self, i=0):
        self.i = i

    def get_PRNGKey(self):
        res = jax.random.PRNGKey(self.i)
        self.i = self.i + 1
        return res


random_handler = RandomHandler()


# %%


def mogpe_with_RFGP_fixed_Omega(X, M=None, Omega_fixed=Omega_fixed, Y=None):
    # M is the number of experts
    # S is the number of spectral frequencies = one half of number of basis functions
    S = Omega_fixed.shape[0]

    with numpyro.plate("M", M, dim=-2):
        var_mu = numpyro.sample("kernel_var_mu_ex", dist.HalfNormal(1.0))
        # noise_mu = numpyro.sample("kernel_noise_mu_ex",dist.InverseGamma(5.0, 5.0))

        var_logstd = numpyro.sample("kernel_var_logstd_ex", dist.HalfNormal(1.0))
        # noise_logstd = numpyro.sample("kernel_noise_logstd_ex",dist.InverseGamma(5.0, 5.0))

        with numpyro.plate("2S", 2 * S):
            theta_mu_ex = numpyro.sample(
                "theta_mu_ex", dist.Normal(loc=0.0, scale=1.0)
            )  # I'm not considering signal power here!
            theta_logstd_ex = numpyro.sample(
                "theta_logstd_ex", dist.Normal(loc=0.0, scale=1.0)
            )  # I'm not considering signal power here!
        with numpyro.plate("DIM", DIM):
            lengthscale_mu_ex = numpyro.sample("ell_mu_ex", dist.HalfNormal(scale=0.1))
            lengthscale_logstd_ex = numpyro.sample(
                "ell_logstd_ex", dist.HalfNormal(scale=1)
            )

    #### MU EX
    vdivide = jax.vmap(lambda x, y: jnp.divide(x, y), (None, 0))
    Omega_mu_ex = vdivide(Omega_fixed, lengthscale_mu_ex)  # shape = (M,S,DIM)
    ########
    # assert Omega_ex.shape == (M,S,DIM)
    ola = X @ jnp.transpose(
        Omega_mu_ex, axes=(0, 2, 1)
    )  # this is batch matmul: (Ndata,DIM) x (M,DIM,S) = (M,Ndata,S)
    # assert ola.shape == (M,N,S)
    Phi_mu_ex = jnp.concatenate(
        [jnp.sin(ola), jnp.cos(ola)], axis=2
    )  # tiene shape = (M, Ndata, 2*S)
    Phi_mu_ex = 1 / jnp.sqrt(S) * Phi_mu_ex

    #### LOGSTD EX
    Omega_logstd_ex = vdivide(Omega_fixed, lengthscale_logstd_ex)
    ola = X @ jnp.transpose(
        Omega_logstd_ex, axes=(0, 2, 1)
    )  # this is batch matmul: (Ndata,DIM) x (M,DIM,S) = (M,Ndata,S)
    # assert ola.shape == (M,N,S)
    Phi_logstd_ex = jnp.concatenate(
        [jnp.sin(ola), jnp.cos(ola)], axis=2
    )  # tiene shape = (M, Ndata, 2*S)
    Phi_logstd_ex = 1 / jnp.sqrt(S) * Phi_logstd_ex

    assert var_mu.shape == (M, 1)
    assert theta_mu_ex.shape == (M, 2 * S)

    theta_mu_ex = jnp.tile(jnp.sqrt(var_mu), 2 * S) * theta_mu_ex
    theta_logstd_ex = jnp.tile(jnp.sqrt(var_logstd), 2 * S) * theta_logstd_ex

    mu_ex = numpyro.deterministic(
        "mu_ex", matmul_vmapped(Phi_mu_ex, theta_mu_ex)
    )  # shape = (M,Ndata)
    logstd_ex = numpyro.deterministic(
        "logstd_ex", matmul_vmapped(Phi_logstd_ex, theta_logstd_ex)
    )  # shape = (M,Ndata)
    std_ex = numpyro.deterministic("std_ex", jnp.exp(logstd_ex))  # shape = (M,Ndata)

    ################################
    # GP for unconstrained weights #
    ################################

    with numpyro.plate("M1", M - 1, dim=-2):
        var_w = numpyro.sample("kernel_var_w", dist.HalfNormal(1.0))
        with numpyro.plate("2S", 2 * S):
            theta_w = numpyro.sample(
                "theta_w",
                dist.Normal(
                    loc=0.0, scale=1.0
                ),  # I'm not considering signal power here!
            )
        with numpyro.plate("DIM", DIM):
            lengthscale_w = numpyro.sample("ell_w", dist.HalfNormal(scale=1))

    assert var_w.shape == (M - 1, 1)
    assert theta_w.shape == (M - 1, 2 * S)

    #### W_UN
    Omega_w = vdivide(Omega_fixed, lengthscale_w)
    ola = X @ jnp.transpose(Omega_w, axes=(0, 2, 1))
    Phi_w = jnp.concatenate(
        [jnp.sin(ola), jnp.cos(ola)], axis=2
    )  # tiene shape = (M, Ndata, 2*S)
    Phi_w = 1 / jnp.sqrt(S) * Phi_w

    theta_w = jnp.tile(jnp.sqrt(var_w), 2 * S) * theta_w
    w_un = matmul_vmapped(Phi_w, theta_w)  # shape = (M-1,Ndata)
    w_un = jnp.vstack([w_un, jnp.zeros(X.shape[0])])  # shape = (M,Ndata)

    log_w = jax.nn.log_softmax(w_un, axis=0)  # shape = (M,Ndata)

    # compute log-factor
    Y_rep = jnp.tile(jnp.reshape(Y, (-1, 1)), M)
    lpd_point = jax.scipy.stats.norm.logpdf(Y_rep, loc=mu_ex.T, scale=std_ex.T)
    logp = jax.nn.logsumexp(lpd_point + log_w.T, axis=1)
    numpyro.deterministic(
        "lpd_point", logp
    )  # I store the "point-wise" evaluation of the log-likelihood
    numpyro.deterministic("w", jnp.exp(log_w).T)
    numpyro.factor("logp", jnp.sum(logp))  




# %%  SVI SVI SVI SVI SVI



svi= SVI(
        mogpe_with_RFGP_fixed_Omega,
        AutoDelta(mogpe_with_RFGP_fixed_Omega, 
                  init_loc_fn = numpyro.infer.initialization.init_to_median),
        # optim = numpyro.optim.Minimize(),  # Using lbfgs instead of adam
        optim=chain(clip(10.0), adam(0.01)),
        loss=Trace_ELBO(),
    )


print("starting SVI...")
res = svi.run(
    random_handler.get_PRNGKey(),
    # 5,
    5000,  # these many iterations when using adam optimizer
    X=X_train,
    Y=y_train,
    M=M_models,
)

print('SVI done')


params = res.params
guide = AutoDelta(mogpe_with_RFGP_fixed_Omega)
# use guide to make predictive
dist_posterior_predictive = Predictive(model=mogpe_with_RFGP_fixed_Omega, 
                                       guide=guide, params=params, num_samples=1)
# these are not actual samples but just the MAP estimates (since we are using AutoDelta)
samples_posterior_predictive = dist_posterior_predictive(random_handler.get_PRNGKey(), 
                                                         X=X_test, 
                                                         Y=y_test, 
                                                         M=M_models,
                                                         )
ymu_tst_svi = samples_posterior_predictive["mu_ex"]

lpd_pogpe_svi_test = samples_posterior_predictive["lpd_point"].mean()
mse_pogpe_svi_test = np.mean((ymu_tst_svi-y_test)**2)

# these are not actual samples but just the MAP estimates (since we are using AutoDelta)
samples_posterior_predictive = dist_posterior_predictive(random_handler.get_PRNGKey(), 
                                                         X=X_train, 
                                                         Y=None, 
                                                         M=M_models,
                                                         )
ymu_tr_svi = samples_posterior_predictive["mean_fused"].mean(0)

lpd_pogpe_svi_train = samples_posterior_predictive["lpd_point"].mean()
mse_pogpe_svi_train = np.mean((ymu_tr_svi-y_train)**2)




# %% MCMC MCMC MCMC MCMC

miMCMC = NUTS(
    mogpe_with_RFGP_fixed_Omega,
    max_tree_depth=(10, 5),
    find_heuristic_step_size=True,
    init_strategy=numpyro.infer.initialization.init_to_median,
)

fer = MCMC(
    miMCMC,
    num_chains=4,
    num_samples=500,
    num_warmup=500,
    thinning=1,
    progress_bar=True,
    chain_method="parallel",
)
fer.run(
    random_handler.get_PRNGKey(),
    X_train,
    Y=y_train,
    M=M_models,
)

miSamples_mogpe = fer.get_samples()


lpd_mogpe_mcmc_train = jax.nn.logsumexp(miSamples_mogpe["lpd_point"], axis=0) - np.log(
    miSamples_mogpe["lpd_point"].shape[0]
)
predict = numpyro.infer.Predictive(mogpe_with_RFGP_fixed_Omega, miSamples_mogpe)
preds = predict(
    random_handler.get_PRNGKey(),
    X_test,
    Y=y_test,
    M=M_models,
)
lpd_mogpe_mcmc_test = jax.nn.logsumexp(preds["lpd_point"], axis=0) - np.log(
    preds["lpd_point"].shape[0]
)



# %%

np.savez(
    f"results_{args.dataset}_M_{M_models}_S_{S}_i_{i_split}",
    lpd_mogpe_mcmc_train=lpd_mogpe_mcmc_train,
    lpd_mogpe_mcmc_test=lpd_mogpe_mcmc_test,
    mse_mogpe_mcmc_train=mse_mogpe_mcmc_train,
    mse_mogpe_mcmc_test=mse_mogpe_mcmc_test,
    lpd_mogpe_svi_train=lpd_mogpe_svi_train,
    lpd_mogpe_svi_test=lpd_mogpe_svi_test,
    mse_mogpe_svi_train=mse_mogpe_svi_train,
    mse_mogpe_svi_test=mse_mogpe_svi_test,
    ymu_tst_mcmc = ymu_tst_mogpe,
    ymu_tst_svi = ymu_tst_svi,
    y_test = y_test
)
