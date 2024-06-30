# import sys
# import os
# # Add the modules directory to the system path (This is if we want to get rid of modules. when importing)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import numpy as np  # Add this line
import matplotlib.pyplot as plt

from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.prediction_storage import store_predictions
from modules.fusion_methods import compute_neg_log_like, product_fusion, phs, bhs, train_stacking, predict_stacking
from modules.model_training import train_and_predict_single_gp


# Parameters
# dataset_name = 'concrete'  
dataset_name = 'yacht'  
split = 9
n_experts = 5
n_points_per_split = 5
kappa = 2    # noise = np.var(y_train)/kappa**2  ; kappa \in [2,100]
lambdaa = 1   # lengthscale = np.std(X_train,1)/lambdaa ; lambdaa \in [1,10]

# Load and normalize data
X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name, split)


# Split dataset and create validation set
n_data_per_expert = X_train.shape[0] // n_experts
splits = split_dataset(X_train, y_train, n_splits=n_experts, split_size=n_data_per_expert, with_replacement=False)
X_val, y_val = create_validation_set(splits, n_points_per_split)

# Store predictions
mu_preds_val, std_preds_val, mu_preds_test, std_preds_test = store_predictions(splits, X_val, X_test, kappa, lambdaa)

# Compute negative log likelihood for experts
nlpd_experts = compute_neg_log_like(mu_preds_test, std_preds_test, y_test)

# ---------  Single GP using all training data ----- #
test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_val, kappa, lambdaa)
nlpd_single_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)


# ----------- Fusion using gpoe --------- #
# mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test, std_preds_test, y_train, kappa)
mus_gpoe, stds_gpoe, w_gpoe = product_fusion(mu_preds_test, std_preds_test, splits, kappa)  # esto seria lo correcto porque cada GP usa su propio y_train para inicializar el noise y el outputscale
nlpd_gpoe = compute_neg_log_like(mus_gpoe, stds_gpoe, y_test)

# Output results
print("NLPD Experts: ", nlpd_experts)
print("NLPD Single GP: ", nlpd_single_gp)
print("NLPD GPOE: ", nlpd_gpoe)



# ---------- PHS training and prediction --------- #
samples_phs = train_stacking(
    model=phs,
    X_val=X_val,
    mu_preds_val=mu_preds_val,
    std_preds_val=std_preds_val,
    y_val=y_val,
)

preds_phs, lpd_phs_test = predict_stacking(
    model=phs,
    samples=samples_phs,
    X_val=X_val,
    X_test=X_test,
    mu_preds_test=mu_preds_test,
    std_preds_test=std_preds_test,
    y_test=y_test,
    prior_mean=lambda x: -np.log(mu_preds_test.shape[1]) * np.ones(x.shape[0]),
)

print("NLPD PHS: ", -lpd_phs_test.mean())



# ------------  BHS training and prediction -------- #
samples_bhs = train_stacking(
    model=bhs,
    X_val=X_val,
    mu_preds_val=mu_preds_val,
    std_preds_val=std_preds_val,
    y_val=y_val,
)

preds_bhs, lpd_bhs_test = predict_stacking(
    model=bhs,
    samples=samples_bhs,
    X_val=X_val,
    X_test=X_test,
    mu_preds_test=mu_preds_test,
    std_preds_test=std_preds_test,
    y_test=y_test,
    prior_mean=lambda x: -np.log(mu_preds_test.shape[1]) * np.ones(x.shape[0]),
)

print("NLPD BHS: ", -lpd_bhs_test.mean())



plt.figure()
for i in range(w_gpoe.shape[1]):
    plt.plot(w_gpoe[:, i])
plt.title("gPoEs")


plt.figure()
for i in range(n_experts):
    plt.plot(preds_phs["w"].mean(0)[i,:])
# plt.show()
plt.title("PHS") 


plt.figure()
for i in range(n_experts):
    plt.plot(preds_bhs["w"].mean(0)[:, i])
plt.title("BHS")


plt.show()


