import numpy as np
import scipy
from .phs import phs
from .bhs import bhs
from .common import train_stacking, predict_stacking
from .common import train_stacking_with_svi, predict_stacking_with_rff
from .common import predict_stacking_without_noise


def compute_neg_log_like(mus, stds, y_test):
    if mus.ndim==1:  # if we are passed vectors instead of one-column matrices
        mus = mus.reshape(-1,1)
        stds = stds.reshape(-1,1)

    negloglik = np.zeros((y_test.shape[0], mus.shape[1]))
    for i in range(mus.shape[1]):
        negloglik[:, i] = -1.0 * scipy.stats.norm.logpdf(y_test.squeeze(), mus[:, i].squeeze(), stds[:, i].squeeze())
    return negloglik.mean(0)


def normalize_weights(weights):
    return weights / np.sum(weights, axis=1, keepdims=True)

def product_fusion(mus, stds, stds_prior=None, 
                   weighting="entropy", method="gPoE", 
                   normalize=True,softmax=False, power=15):
    # mus        is num_test x num_experts
    # stds       is num_test x num_experts
    # stds_prior is num_test x num_experts

    assert mus.ndim == 2
    assert stds.ndim == 2
    if stds_prior is not None:
        assert stds_prior.ndim == 2

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

    assert weights.shape == mus.shape    

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
        if "kernel_noise" in samples.keys():
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
        else:
            preds, lpd_test = predict_stacking_without_noise(
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
    else:
        raise ValueError("GP model not valid.")

    return preds, lpd_test





class TrainingConfig:
    def __init__(self, inference_method="mcmc", gp_method="vanilla", 
                 parallel_mcmc=False, guide_svi=None, 
                 show_progress=False, show_summary=False,
                 num_warmup=100, num_samples=100,num_chains=4,  # for NUTS
                 lr_svi=0.005, training_iter_svi=3000,          # for SVI
                 ): 
        
        self.inference_method = inference_method
        self.gp_method = gp_method
        self.parallel_mcmc = parallel_mcmc
        self.guide_svi = guide_svi
        self.show_progress = show_progress
        self.show_summary = show_summary
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.lr_svi = lr_svi
        self.training_iter_svi = training_iter_svi



def train_and_predict_fusion_method_new(model,data,config):

    ''' data is a dictionary:
    data = {
        "X_val": X_val,
        "mu_preds_val": mu_preds_val,
        "std_preds_val": std_preds_val,
        "y_val": y_val,
        "X_test": X_test,
        "mu_preds_test": mu_preds_test,
        "std_preds_test": std_preds_test,
        "y_test": y_test
        }

        config is a TrainingConfig object
    '''
    
    if config.inference_method == "mcmc":
        samples = train_stacking(
            model=model,
            X_val=data["X_val"],
            mu_preds_val=data["mu_preds_val"],
            std_preds_val=data["std_preds_val"],
            y_val=data["y_val"],
            show_progress=config.show_progress,
            parallel=config.parallel_mcmc,
            show_summary=config.show_summary,
            num_warmup=config.num_warmup,
            num_samples=config.num_samples,
            num_chains=config.num_chains,
        )
    elif config.inference_method == "svi":
        samples = train_stacking_with_svi(
            model=model,
            X_val=data["X_val"],
            mu_preds_val=data["mu_preds_val"],
            std_preds_val=data["std_preds_val"],
            y_val=data["y_val"],
            guide_svi=config.guide_svi,
            progress_bar=config.show_progress,
            learning_rate=config.lr_svi,
            training_iter=config.training_iter_svi,
        )

    # Predict using the trained fusion model
    if config.gp_method == "vanilla":
        if "kernel_noise" in samples.keys():
            preds, lpd_test = predict_stacking(
                model=model,
                samples=samples,
                X_val=data["X_val"],
                X_test=data["X_test"],
                mu_preds_test=data["mu_preds_test"],
                std_preds_test=data["std_preds_test"],
                y_test=data["y_test"],
                prior_mean=lambda x: -np.log(data["mu_preds_test"].shape[1]) * np.ones(x.shape[0]),
            )
        else:
            preds, lpd_test = predict_stacking_without_noise(
                model=model,
                samples=samples,
                X_val=data["X_val"],
                X_test=data["X_test"],
                mu_preds_test=data["mu_preds_test"],
                std_preds_test=data["std_preds_test"],
                y_test=data["y_test"],
                prior_mean=lambda x: -np.log(data["mu_preds_test"].shape[1]) * np.ones(x.shape[0]),
            )
    elif config.gp_method == "rff":
        preds, lpd_test = predict_stacking_with_rff(
            model=model, 
            samples=samples, 
            X_test=data["X_test"], 
            mu_preds_test=data["mu_preds_test"], 
            std_preds_test=data["std_preds_test"], 
            y_test=data["y_test"]
        )
    else:
        raise ValueError("GP model not valid.")

    return preds, lpd_test

