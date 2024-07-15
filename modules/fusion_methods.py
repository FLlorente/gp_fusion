import numpy as np
import scipy
from .phs import phs
from .bhs import bhs
from .common import train_stacking, predict_stacking
from .common import train_stacking_with_svi, predict_stacking_with_rff


def compute_neg_log_like(mus, stds, y_test):
    if mus.ndim==1:  # if we are passed vectors instead of one-column matrices
        mus = mus.reshape(-1,1)
        stds = stds.reshape(-1,1)

    negloglik = np.zeros((y_test.shape[0], mus.shape[1]))
    for i in range(mus.shape[1]):
        negloglik[:, i] = -1.0 * scipy.stats.norm.logpdf(y_test, mus[:, i], stds[:, i])
    return negloglik.mean(0)


def normalize_weights(weights):
    return weights / np.sum(weights, axis=1, keepdims=True)

def product_fusion(mus, stds, stds_prior=None, 
                   weighting="entropy", method="gPoE", 
                   normalize=True,softmax=False, power=15):
    # mus        is num_test x num_experts
    # stds       is num_test x num_experts
    # stds_prior is num_test x num_experts

    # Number of experts
    M = mus.shape[1]

    # Compute the weights matrix
    if weighting == "entropy":
        weights = 0.5 * (np.log(stds_prior**2) - np.log(stds**2))  
        if softmax:
            weights = np.exp(power * weights)
    elif weighting == "uniform":
        weights = np.ones_like(mus) / M
    elif weighting == "variance":
        weights = np.exp(-power * stds**2)
    elif weighting == "no-weights":
        weights = np.ones_like(mus)
        

    # Normalize the weights if required
    if normalize:
        weights = normalize_weights(weights)

    # Compute precisions
    precs = 1 / stds**2

    # Compute fused precision and mean based on the method
    if method == "PoE":
        prec_fused = np.sum(precs, axis=1, keepdims=True)  # Sum over experts

    elif method == "gPoE":
        prec_fused = np.sum(weights * precs, axis=1, keepdims=True)  # Weighted sum over experts

    elif method == "BCM":
        prior_var = stds_prior[0, 0]**2 # Prior variance
        prec_fused = np.sum(precs, axis=1, keepdims=True) + (1 - M) / prior_var  # Sum plus correction

    elif method == "rBCM":
        prior_var = stds_prior[0, 0]**2 # Prior variance
        prec_fused = np.sum(weights * precs, axis=1, keepdims=True) + (1 - np.sum(weights, axis=1, keepdims=True)) / prior_var  # Weighted sum plus correction

    elif method == "bar":
        var_fused = np.sum(weights * stds**2, axis=1, keepdims=True)  # Weighted average of variances
        mean_fused = np.sum(weights * mus, axis=1, keepdims=True)  # Weighted average of means
        return mean_fused, np.sqrt(var_fused), weights

    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute fused mean
    mean_fused = np.sum(weights * mus * precs, axis=1, keepdims=True) / prec_fused

    return mean_fused, 1 / np.sqrt(prec_fused), weights

def train_and_predict_fusion_method(model, 
                                    X_val, 
                                    mu_preds_val, 
                                    std_preds_val, 
                                    y_val, 
                                    X_test, 
                                    mu_preds_test, 
                                    std_preds_test, 
                                    y_test,
                                    inference_method = "mcmc",
                                    gp_method = "vanilla",
                                    parallel_mcmc = False,
                                    guide_svi = None,
                                    show_progress = False,
                                    show_summary = False,
                                    ):
    
    if inference_method == "mcmc":
        # Train the fusion model
        samples = train_stacking(
            model=model,
            X_val=X_val,
            mu_preds_val=mu_preds_val,
            std_preds_val=std_preds_val,
            y_val=y_val,
            show_progress=show_progress,
            parallel=parallel_mcmc,
            show_summary=show_summary,
        )
    elif inference_method == "svi":
        samples = train_stacking_with_svi(
            model=model,
            X_val=X_val,
            mu_preds_val=mu_preds_val,
            std_preds_val=std_preds_val,
            y_val=y_val,
            guide_svi=guide_svi,
            progress_bar=show_progress,
        )

    # Predict using the trained fusion model
    if gp_method == "vanilla":
        preds, lpd_test = predict_stacking(
            model=model,
            samples=samples,
            X_val=X_val,
            X_test=X_test,
            mu_preds_test=mu_preds_test,
            std_preds_test=std_preds_test,
            y_test=y_test,
            prior_mean=lambda x: -np.log(mu_preds_test.shape[1]) * np.ones(x.shape[0]),
        )
    elif gp_method == "rff":
        preds, lpd_test = predict_stacking_with_rff(
                                                    model, 
                                                    samples, 
                                                    X_test, 
                                                    mu_preds_test, 
                                                    std_preds_test, 
                                                    y_test
                                                    )

    return preds, lpd_test