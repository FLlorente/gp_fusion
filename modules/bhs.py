import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from .common import vmap_SE_kernel

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
