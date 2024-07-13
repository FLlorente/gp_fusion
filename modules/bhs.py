import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from .common import vmap_SE_kernel, matmul_vmapped, vdivide



def bhs(X, mu_preds, std_preds, y_val=None):
    N, M = mu_preds.shape

    assert mu_preds.shape == std_preds.shape

    ######################
    # GP for log weights #
    ######################
    with numpyro.plate("M", M):
        kernel_var = numpyro.sample("kernel_var", dist.HalfNormal(1.0))
        kernel_length = numpyro.sample(
            "kernel_length", 
            # dist.InverseGamma(5.0, 5.0),
            dist.HalfNormal(1.0),
            )
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



'''
BHS with RFF-GPs
'''


def bhs_with_rff(X, mu_preds, std_preds, y_val=None, S=50):
    N, M = mu_preds.shape
    DIM = X.shape[1]

    assert mu_preds.shape == std_preds.shape

    Omega = jax.random.normal(jax.random.PRNGKey(0), (S, DIM))
    # with numpyro.plate("Omega_rows", S, dim=0):  # this is if we want to learn the location of the freqs too
    #     with numpyro.plate("Omega_cols", DIM, dim=1):
    #         Omega = numpyro.sample("Omega",dist.Normal(loc=0.0, scale=1.0))



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
            lengthscale_w = numpyro.sample("ell_w", dist.HalfNormal(scale=1.0))

    assert var_w.shape == (M - 1, 1)
    assert theta_w.shape == (M - 1, 2 * S)

    #### W_UN
    Omega_w = vdivide(Omega, lengthscale_w)
    ola = X @ jnp.transpose(Omega_w, axes=(0, 2, 1))
    Phi_w = jnp.concatenate(
        [jnp.sin(ola), jnp.cos(ola)], axis=2
    )  # tiene shape = (M, Ndata, 2*S)
    Phi_w = Phi_w / jnp.sqrt(S)

    theta_w = jnp.tile(jnp.sqrt(var_w), 2 * S) * theta_w
    w_un = matmul_vmapped(Phi_w, theta_w)  # shape = (M-1,Ndata)
    w_un = jnp.vstack([w_un, jnp.zeros(X.shape[0])])  # shape = (M,Ndata)

    log_w = jax.nn.log_softmax(w_un.T, axis=1)  # shape = (Ndata,M)

    #################
    # Fuse with BHS #
    #################
    y_val_rep = jnp.tile(jnp.reshape(y_val, (-1, 1)), M)
    lpd_point = jax.scipy.stats.norm.logpdf(y_val_rep, loc=mu_preds, scale=std_preds)

    logp = jax.nn.logsumexp(lpd_point + log_w, axis=1)
    numpyro.deterministic("lpd_point", logp)
    numpyro.deterministic("w", jnp.exp(log_w))
    numpyro.factor("logp", jnp.sum(logp))


