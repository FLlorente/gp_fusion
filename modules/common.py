import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta, AutoNormal
from optax import adam, chain, clip

# Vectorized matrix multiplication
matmul_vmapped = jax.vmap(lambda A,B: (A @ B).squeeze(), in_axes=(0, 0))

# Vectorized division
vdivide = jax.vmap(lambda x, y: jnp.divide(x, y), (None, 0))

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
    return mean, jnp.sqrt(jnp.diag(K))

vmapped_pred_with_mean = jax.vmap(
    predict_with_mean, in_axes=(None, None, 0, None, 0, 0, 0, None, None)
)

vmapped_pred_with_mean = jax.vmap(
    vmapped_pred_with_mean,
    in_axes=(None, None, 1, None, 1, 1, 1, None, None),
)


def train_stacking(model, X_val, mu_preds_val, std_preds_val, y_val, 
                   parallel=False, show_progress=False,show_summary=False,
                   num_warmup=100, num_samples=100, num_chains=4):

    if parallel:
        numpyro.set_host_device_count(4)
        numpyro.set_platform("cpu")
        numpyro.enable_x64()

        chain_method = "parallel"
    else:
        chain_method = "sequential"

    mcmc = MCMC(
        NUTS(model, 
            init_strategy=numpyro.infer.initialization.init_to_median,
        ),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=show_progress,
        chain_method=chain_method,
    )

    mcmc.run(random.PRNGKey(0), 
             X_val,   
             mu_preds_val, 
             std_preds_val, 
             y_val=y_val, 
    )
    
    if show_summary:
        mcmc.print_summary()
    samples = mcmc.get_samples()

    return samples

def predict_stacking(model, samples, X_val, X_test, mu_preds_test, std_preds_test, y_test, prior_mean = lambda x: jnp.zeros(x.shape[0])):
    res = vmapped_pred_with_mean(
        random.PRNGKey(0),  # this one is not being use since we only extract the posterior mean
        X_val,
        samples["w_un"],
        X_test, # TEST DATA
        samples["kernel_var"],
        samples["kernel_length"],
        samples["kernel_noise"],
        SE_kernel,
        prior_mean,
    )

    w_un_samples = jnp.asarray(res[0])  # we could have used a random sample instead of the posterior mean
    pred_samples = {"w_un": jnp.transpose(w_un_samples, (1, 0, 2))}

    predictive = Predictive(model, pred_samples)
    pred_samples = predictive(
        random.PRNGKey(0),   # this one doesn't matter since we only want to extract the lpd values
        X=X_test, # TEST DATA
        mu_preds=mu_preds_test,  # TEST DATA
        std_preds=std_preds_test, # TEST DATA
        y_val = y_test, # TEST DATA
    )

    lpd_test = jax.nn.logsumexp(pred_samples["lpd_point"], axis=0) - jnp.log(pred_samples["lpd_point"].shape[0])

    return pred_samples, lpd_test



def train_stacking_with_svi(model, X_val, mu_preds_val, std_preds_val, y_val,
                           guide_svi="map", progress_bar=False,
                           learning_rate=0.005, training_iter = 3000):
    if guide_svi == "map":
        svi= SVI(model,
            AutoDelta(model, 
                    init_loc_fn = numpyro.infer.initialization.init_to_median,
                    ),
            optim=chain(clip(10.0), adam(learning_rate)),
            loss=Trace_ELBO(),
        )
    elif guide_svi=="normal":
        svi= SVI(model,
            AutoNormal(model, 
                    init_loc_fn = numpyro.infer.initialization.init_to_median,
                    ),
            optim=chain(clip(10.0), adam(learning_rate)),
            loss=Trace_ELBO(),
        )

    results = svi.run(
        random.PRNGKey(0),
        training_iter,  # for adam
        X_val,   
        mu_preds_val, 
        std_preds_val, 
        y_val=y_val,
        progress_bar=progress_bar,
    )

    params = results.params

    if guide_svi == "map":
        guide = AutoDelta(model)
        predictive = Predictive(guide, params=params, num_samples=1)  # more than 1 is pointless
    elif guide_svi == "normal":
        guide = AutoNormal(model)
        predictive = Predictive(guide, params=params, num_samples=400)

    samples = predictive(random.PRNGKey(0),X_val,mu_preds_val,std_preds_val,y_val)  # these are samples from the guide! See https://github.com/pyro-ppl/numpyro/issues/1309

    return samples


def predict_stacking_with_rff(model, samples, X_test, mu_preds_test, std_preds_test, y_test):
    predictive = Predictive(model, samples)
    preds = predictive(             # these are the samples from the posterior predictive!
        jax.random.PRNGKey(0),
        X=X_test,
        mu_preds=mu_preds_test,
        std_preds=std_preds_test,
        y_val=y_test,
    )

    lpd = jnp.mean(
        jax.nn.logsumexp(preds["lpd_point"], axis=0) - jnp.log(preds["lpd_point"].shape[0])
    )

    return preds, lpd



vmap_SE_kernel_fer = jax.vmap(SE_kernel, in_axes=(None, None, 0, 0, None)) # without noise

vmapped_pred_with_mean_fer = jax.vmap(
    predict_with_mean, in_axes=(None, None, 0, None, 0, 0, None, None, None) # without noise
)

vmapped_pred_with_mean_fer = jax.vmap(
    vmapped_pred_with_mean_fer,
    in_axes=(None, None, 1, None, 1, 1, None, None, None),  # without noise
)

def predict_stacking_without_noise(model, samples, X_val, X_test, mu_preds_test, std_preds_test, y_test, prior_mean = lambda x: jnp.zeros(x.shape[0])):
    
    if "kernel_noise" in samples.keys():
        raise ValueError("Use this function with phs_without_noise/bhs_without_noise only.")
    
    res = vmapped_pred_with_mean_fer(
        random.PRNGKey(0),
        X_val,
        samples["w_un"],
        X_test, # TEST DATA
        samples["kernel_var"],
        samples["kernel_length"],
        0,  # without noise
        SE_kernel,
        prior_mean,
    )

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

    lpd_test = jax.nn.logsumexp(pred_samples["lpd_point"], axis=0) - jnp.log(pred_samples["lpd_point"].shape[0])

    return pred_samples, lpd_test