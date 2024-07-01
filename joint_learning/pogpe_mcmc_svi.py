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
# np.random.seed(0)
# Omega_fixed= jnp.asarray(np.random.randn(S*DIM)).reshape(S,DIM)
# %%

def pogpe_with_RFGP_fixed_Omega(X, M=None, Omega_fixed = Omega_fixed, Y=None, y_test=None):              # M is the number of experts
    N = X.shape[0]
    DIM = X.shape[1]
    S = Omega_fixed.shape[0]    # number of spectral frequencies

    #########################
    # RF-GP for the experts #
    #########################
    with numpyro.plate("M", M, dim=-2):
        var_mu_ex = numpyro.sample("kernel_var_exp_mean", dist.HalfNormal(1.0))
        # noise_mu_ex = numpyro.sample("kernel_noise_exp_mean", dist.InverseGamma(5.0, 5.0))
        std_ex_un = numpyro.sample('std_ex_un', dist.InverseGamma(5.0, 5.0))
        std_ex = numpyro.deterministic('std_ex', jnp.tile(jnp.reshape(std_ex_un, (-1, 1)), N))

        with numpyro.plate('2S', 2*S):
            theta_mu_ex = numpyro.sample("theta_mu_ex", dist.Normal(loc=0.0, 
                                                        scale=1.0)) # I'm not considering signal power here!
            
        with  numpyro.plate('DIM', DIM):
            lengthscale_ex = numpyro.sample('ell_ex', dist.HalfNormal(scale = 1))

            
    assert lengthscale_ex.shape == (M,DIM)
    assert Omega_fixed.shape == (S,DIM)

    #### OMEGA FIXED
    vdivide =  jax.vmap(lambda x, y: jnp.divide(x, y), (None, 0))                                                                                                                            
    Omega_ex = vdivide( Omega_fixed, lengthscale_ex)     # shape = (M,S,DIM)
    ########
    assert Omega_ex.shape == (M,S,DIM)
    ola = X @ jnp.transpose(Omega_ex, axes=(0,2,1))     # this is batch matmul: (Ndata,DIM) x (M,DIM,S) = (M,Ndata,S)
    #assert ola.shape == (M,N,S)
    Phi_ex = jnp.concatenate([jnp.sin(ola), jnp.cos(ola)], axis=2)  # tiene shape = (M, Ndata, 2*S)
    Phi_ex = 1/jnp.sqrt(S)*Phi_ex  # se me habia olvidado dividir entre jnp.sqrt(S)
    assert Phi_ex.shape == (M, N, 2*S)
    assert std_ex.shape == (M,N)
    assert theta_mu_ex.shape == (M,2*S)

    # theta_mu_ex = jnp.tile(jnp.sqrt(var_mu_ex+ noise_mu_ex),2*S) * theta_mu_ex
    theta_mu_ex = jnp.tile(jnp.sqrt(var_mu_ex),2*S) * theta_mu_ex
    mu_ex = numpyro.deterministic("mu_ex", matmul_vmapped(Phi_ex,theta_mu_ex) )                 # shape = (M,Ndata)
    assert mu_ex.shape == (M,N)



    #########################
    # RF-GP for log weights #
    #########################
    with numpyro.plate("M", M, dim=-2):
        # set uninformative log-normal priors on our three kernel hyperparameters
        var_logw = numpyro.sample("kernel_var_logw", dist.HalfNormal(1.0))
        # noise_logw = numpyro.sample("kernel_noise_logw", dist.InverseGamma(5.0, 5.0))

        with numpyro.plate('2S', 2*S):
            theta_logw = numpyro.sample("theta_logw", dist.Normal(loc=0.0, 
                                                        scale=1.0)) # I'm not considering signal power here!
            
        with numpyro.plate("DIM", DIM):
            lengthscale_logw = numpyro.sample('ell_logw', dist.HalfNormal(scale = 1))

    assert lengthscale_logw.shape == (M,DIM)

    # theta_logw = jnp.tile(jnp.sqrt(var_logw+ noise_logw),2*S) * theta_logw
    theta_logw = jnp.tile(jnp.sqrt(var_logw),2*S) * theta_logw
    assert theta_logw.shape == (M,2*S)


    Omega_logw = vdivide( Omega_fixed, lengthscale_logw)     # shape = (M,S,DIM)
    ola = X @ jnp.transpose(Omega_logw, axes=(0,2,1))     # this is batch matmul: (Ndata,DIM) x (M,DIM,S) = (M,Ndata,S)

    Phi_logw = jnp.concatenate([jnp.sin(ola), jnp.cos(ola)], axis=2)  # tiene shape = (M, Ndata, 2*S)
    Phi_logw = 1/jnp.sqrt(S)*Phi_logw  # se me habia olvidado dividir entre jnp.sqrt(S)

    logw = numpyro.deterministic("logw", matmul_vmapped(Phi_logw, theta_logw) - jnp.log(M))   # PRIOR MEAN FUNCTION!
    assert logw.shape == (M,N)


    ################################################
    # Fuse with generalized multiplicative pooling #
    ################################################
    w = numpyro.deterministic("w", jnp.exp(logw) )                 # shape = (M,Ndata)
    w  = w.T
    tau_ex = 1. / std_ex


    tau_fused = numpyro.deterministic(
        "tau_fused", jnp.einsum("nm,mn->m", tau_ex, w)
    )  # N,
    assert tau_fused.shape == (N,)

    mu_fused = numpyro.deterministic(
        "mean_fused", jnp.einsum("nm,nm,mn->m", tau_ex, mu_ex, w) / tau_fused
    )  # N,
    assert mu_fused.shape == (N,)
    
    std_fused = numpyro.deterministic("std_fused", 1 / jnp.sqrt(tau_fused))
    assert std_fused.shape == (N,)

    numpyro.sample(
        "y_val",
        dist.Normal(loc=jnp.squeeze(mu_fused), scale=jnp.squeeze(std_fused)),
        obs=Y,
    )

    if y_test is not None:
        # compute the lpd of the observations
        numpyro.deterministic("lpd_point", jax.scipy.stats.norm.logpdf(
                jnp.squeeze(y_test), loc=jnp.squeeze(mu_fused), scale=jnp.squeeze(std_fused),
                )
            )




class RandomHandler:
    def __init__(self, i=0):
        self.i = i

    def get_PRNGKey(self):
        res = jax.random.PRNGKey(self.i)
        self.i = self.i + 1
        return res


random_handler = RandomHandler()




# %% Pogpe SVI

svi= SVI(
        pogpe_with_RFGP_fixed_Omega,
        AutoDelta(pogpe_with_RFGP_fixed_Omega, 
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
guide = AutoDelta(pogpe_with_RFGP_fixed_Omega)
# use guide to make predictive
dist_posterior_predictive = Predictive(model=pogpe_with_RFGP_fixed_Omega, 
                                       guide=guide, params=params, num_samples=1)
# these are not actual samples but just the MAP estimates (since we are using AutoDelta)
samples_posterior_predictive = dist_posterior_predictive(random_handler.get_PRNGKey(), 
                                                         X=X_test, 
                                                         Y=None, 
                                                         M=M_models,
                                                         y_test=y_test)
ymu_tst_svi = samples_posterior_predictive["mean_fused"].mean(0)

lpd_pogpe_svi_test = samples_posterior_predictive["lpd_point"].mean()
mse_pogpe_svi_test = np.mean((ymu_tst_svi-y_test)**2)

# these are not actual samples but just the MAP estimates (since we are using AutoDelta)
samples_posterior_predictive = dist_posterior_predictive(random_handler.get_PRNGKey(), 
                                                         X=X_train, 
                                                         Y=None, 
                                                         M=M_models,
                                                         y_test=y_train)
ymu_tr_svi = samples_posterior_predictive["mean_fused"].mean(0)

lpd_pogpe_svi_train = samples_posterior_predictive["lpd_point"].mean()
mse_pogpe_svi_train = np.mean((ymu_tr_svi-y_train)**2)





# %% POGPEEEEE (MCMC)
miMCMC = NUTS(
    pogpe_with_RFGP_fixed_Omega,
    # max_tree_depth=(10, 5),
    max_tree_depth=2,
    # find_heuristic_step_size=True,
    # init_strategy=numpyro.infer.initialization.init_to_median,
    init_strategy = numpyro.infer.initialization.init_to_value(
                      values={
"kernel_var_exp_mean": params['kernel_var_exp_mean_auto_loc'],
"kernel_var_logw": params["kernel_var_logw_auto_loc"],
"ell_ex": params["ell_ex_auto_loc"],
"ell_logw": params["ell_logw_auto_loc"],
"theta_mu_ex": params["theta_mu_ex_auto_loc"],
"theta_logw": params["theta_logw_auto_loc"],
"std_ex_un": params["std_ex_un_auto_loc"]
                          }),
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

print("starting MCMC...")
fer.run(
    random_handler.get_PRNGKey(),
    X_train,
    Y=y_train,
    y_test=y_train,  # this is line is for computing the lpd of y_train
    M=M_models,
)

print('MCMC done!')

miSamples = fer.get_samples()


predict = numpyro.infer.Predictive(pogpe_with_RFGP_fixed_Omega, miSamples)
preds = predict(
    random_handler.get_PRNGKey(),
    X_test,
    Y=None,
    M=M_models,
    y_test=y_test,  # this line is for computing the lpd of y_test
)

# There's no need for predict in the training dataset
# preds_train = predict(
#     random_handler.get_PRNGKey(),
#     X=X_train,
#     Y=None,
#     y_test=y_train,
#     M=M_models,
# )

# lpd_pogpe_mcmc_train = np.mean(
#     jax.nn.logsumexp(preds_train["lpd_point"], axis=0) - np.log(
#     preds_train["lpd_point"].shape[0]
# )
# )

lpd_pogpe_mcmc_train = np.mean(
    jax.nn.logsumexp(miSamples["lpd_point"], axis=0) - np.log(
    miSamples["lpd_point"].shape[0]
)
)

lpd_pogpe_mcmc_test = np.mean(
    jax.nn.logsumexp(preds["lpd_point"], axis=0) - np.log(
    preds["lpd_point"].shape[0]
)
)

ymu_tr_pogpe = miSamples["mean_fused"].mean(0)
ymu_tst_pogpe = preds["mean_fused"].mean(0)

mse_pogpe_mcmc_train = np.mean((ymu_tr_pogpe-y_train)**2)
mse_pogpe_mcmc_test = np.mean((ymu_tst_pogpe-y_test)**2)





# %%

np.savez(
    f"results_{args.dataset}_M_{M_models}_S_{S}_i_{i_split}",
    lpd_pogpe_mcmc_train=lpd_pogpe_mcmc_train,
    lpd_pogpe_mcmc_test=lpd_pogpe_mcmc_test,
    mse_pogpe_mcmc_train=mse_pogpe_mcmc_train,
    mse_pogpe_mcmc_test=mse_pogpe_mcmc_test,
    lpd_pogpe_svi_train=lpd_pogpe_svi_train,
    lpd_pogpe_svi_test=lpd_pogpe_svi_test,
    mse_pogpe_svi_train=mse_pogpe_svi_train,
    mse_pogpe_svi_test=mse_pogpe_svi_test,
    ymu_tst_mcmc = ymu_tst_pogpe,
    ymu_tst_svi = ymu_tst_svi,
    y_test = y_test
)
