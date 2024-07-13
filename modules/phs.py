import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from .common import vmap_SE_kernel, matmul_vmapped, vdivide

def phs(X, mu_preds, std_preds, y_val=None):
    N, M = mu_preds.shape

    assert mu_preds.shape == std_preds.shape

    tau_preds = 1 / std_preds**2

    ######################
    # GP for log weights #
    ######################
    with numpyro.plate("M", M):
        kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(1.0))
        kernel_length = numpyro.sample("kernel_length", 
                                    #    dist.InverseGamma(5.0, 5.0),
                                       dist.HalfNormal(1.0),
                                       )
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




'''
PHS with weights that sum up to 1
'''

def phs_with_normalized_w(X, mu_preds, std_preds, y_val=None):
    N, M = mu_preds.shape

    assert mu_preds.shape == std_preds.shape

    tau_preds = 1 / std_preds**2

    ######################
    # GP for log weights #
    ######################
    with numpyro.plate("M", M):
        kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(1.0))
        kernel_length = numpyro.sample("kernel_length", 
                                    #    dist.InverseGamma(5.0, 5.0),
                                       dist.HalfNormal(1.0),
                                       )
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
    w = numpyro.deterministic("w", jax.nn.softmax(log_w, axis=0))

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





'''
PHS with RFF-GPs
'''


def phs_with_rff(X, mu_preds, std_preds, y_val=None, S = 50):
    N, M = mu_preds.shape
    DIM = X.shape[1]

    Omega = jax.random.normal(jax.random.PRNGKey(0), (S, DIM))
    # with numpyro.plate("Omega_rows", S, dim=0):  # this is if we want to learn the location of the freqs too
    #     with numpyro.plate("Omega_cols", DIM, dim=1):
    #         Omega = numpyro.sample("Omega",dist.Normal(loc=0.0, scale=1.0))

    assert mu_preds.shape == std_preds.shape

    tau_preds = 1 / std_preds**2

    #########################
    # RF-GP for log weights #
    #########################
    with numpyro.plate("M", M, dim=-2):
        # set uninformative log-normal priors on our three kernel hyperparameters
        var_logw = numpyro.sample("kernel_var_logw", dist.HalfNormal(1.0))
        # noise_logw = numpyro.sample("kernel_noise_logw", dist.InverseGamma(5.0, 5.0))

        with numpyro.plate("2S", 2 * S):
            theta_logw = numpyro.sample(
                "theta_logw", dist.Normal(loc=0.0, scale=1.0)
            )  # I'm not considering signal power here!

        with numpyro.plate("DIM", DIM):
            lengthscale_logw = numpyro.sample("ell_logw", dist.HalfNormal(scale=1))

    assert lengthscale_logw.shape == (M, DIM)

    theta_logw = jnp.tile(jnp.sqrt(var_logw), 2 * S) * theta_logw

    Omega_logw = vdivide(Omega, lengthscale_logw)  # shape = (M,S,DIM)
    ola = X @ jnp.transpose(
        Omega_logw, axes=(0, 2, 1)
    )  # this is batch matmul: (Ndata,DIM) x (M,DIM,S) = (M,Ndata,S)

    Phi_logw = jnp.concatenate(
        [jnp.sin(ola), jnp.cos(ola)], axis=2
    )  # tiene shape = (M, Ndata, 2*S)
    Phi_logw = (
        1 / jnp.sqrt(S) * Phi_logw
    )  # se me habia olvidado dividir entre jnp.sqrt(S)

    logw = numpyro.deterministic(
        "w_un", matmul_vmapped(Phi_logw, theta_logw) - jnp.log(M)
    )
    assert logw.shape == (M, N)

    ################################################
    # Fuse with generalized multiplicative pooling #
    ################################################
    w = numpyro.deterministic("w", jnp.exp(logw))
    # w  = w.T

    # print(w.shape, tau_preds.shape)

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

    # compute the lpd of the observations
    numpyro.deterministic(
        "lpd_point",
        jax.scipy.stats.norm.logpdf(
            y_val,
            loc=jnp.squeeze(mu_fused),
            scale=jnp.squeeze(std_fused),
        ),
    )