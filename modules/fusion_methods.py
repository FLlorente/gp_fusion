import numpy as np
import scipy
from .phs import phs
from .bhs import bhs
from .common import train_stacking, predict_stacking


def compute_neg_log_like(mus, stds, y_test):
    negloglik = np.zeros((y_test.shape[0], mus.shape[1]))
    for i in range(mus.shape[1]):
        negloglik[:, i] = -1.0 * scipy.stats.norm.logpdf(y_test, mus[:, i], stds[:, i])
    return negloglik.mean(0)

def product_fusion(mus, stds, splits, kappa):
    prec_fused = np.zeros((mus.shape[0], 1))
    mean_fused = np.zeros((mus.shape[0], 1))
    w_gpoe = np.zeros(mus.shape)

    # To compute the weights, we need to know the prior predictive variance of the experts
    if isinstance(splits, list):
        # Check if the elements of the list are arrays
        if all(isinstance(item, np.ndarray) for (_,item) in splits):
            # print("every expert uses its own prior variance")
            n = len(splits)
            noise = np.zeros(n)
            variance = np.zeros(n)
            for i,(_,y_train) in enumerate(splits):
                # Perform computation on each array
                noise[i] = np.var(y_train) / kappa**2
                variance[i] = np.var(y_train)
    elif isinstance(splits, np.ndarray):
        y_train = splits
        # print("all experts use the same prior variance")
        noise = np.var(y_train) / kappa**2
        variance = np.var(y_train)
            
    
    for n in range(mus.shape[0]):
        weights = 0.5 * (np.log(noise + variance) - np.log(stds[n, :]**2))  # 0.5(log(sig2prior) - log(sig2post))
        weights = weights / np.sum(weights)

        precs = 1 / stds[n, :]**2

        prec_fused[n, :] = weights @ precs
        mean_fused[n, :] = weights @ (mus[n, :] * precs) / prec_fused[n, :]

        w_gpoe[n, :] = weights

    return mean_fused, 1 / np.sqrt(prec_fused), w_gpoe


def train_and_predict_fusion_method(model, X_val, mu_preds_val, std_preds_val, y_val, X_test, mu_preds_test, std_preds_test, y_test):
    # Train the fusion model
    samples = train_stacking(
        model=model,
        X_val=X_val,
        mu_preds_val=mu_preds_val,
        std_preds_val=std_preds_val,
        y_val=y_val,
    )

    # Predict using the trained fusion model
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

    return preds, lpd_test