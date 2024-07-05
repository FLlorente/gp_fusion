# import sys
# import os
# # Add the modules directory to the system path (This is if we want to get rid of modules. when importing)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import numpy as np  # Add this line
import matplotlib.pyplot as plt

from modules.data_handling import load_and_normalize_data, split_dataset, create_validation_set
from modules.prediction_storage import store_predictions
from modules.fusion_methods import compute_neg_log_like, product_fusion, train_stacking, predict_stacking, train_and_predict_fusion_method
from modules.model_training import train_and_predict_single_gp, train_expert, store_predictions_for_experts
from modules.phs import phs, phs_with_normalized_w
from modules.bhs import bhs
from modules.model_training import train_joint_experts_shared_kernel
from modules.model_training import train_variational_gp, predict_variational_gp

# Parameters
# dataset_name = 'concrete'  
# dataset_name = 'yacht'  
dataset_name = 'elevators'
# dataset_name = "airfoil"
split = 1
n_experts = 5
n_points_per_split = 1
kappa = 2    # noise = np.var(y_train)/kappa**2  ; kappa \in [2,100]
lambdaa = 1   # lengthscale = np.std(X_train,1)/lambdaa ; lambdaa \in [1,10]
lr=0.01  # learning rate of adam optimizer for hyperparameter learning
training_iter = 100
num_epochs = 10

# ------------ Load and normalize data --------- #
X_train, y_train, X_test, y_test = load_and_normalize_data(dataset_name, split)



# ---------  Exact GP using all training data ----- #
test_preds, _ = train_and_predict_single_gp(X_train, y_train, X_test, X_test, 
                                            kappa, lambdaa,
                                            lr=lr,
                                            training_iter=training_iter)
nlpd_exact_gp = compute_neg_log_like(test_preds.mean.numpy().reshape(-1, 1), 
                                      np.sqrt(test_preds.variance.numpy().reshape(-1, 1)), y_test)
# Output results
print("NLPD Exact GP: ", nlpd_exact_gp)

# --------- Variational GP -------- #

# Use a subset of the training points as inducing points
np.random.seed(0)
inducing_points = X_train[np.random.choice(X_train.shape[0], 100, replace=False), :]

# Train the Variational GP with mini-batch
model, likelihood = train_variational_gp(X_train, y_train, 
                                         inducing_points, 
                                         kappa=kappa,lambdaa=lambdaa,
                                         learning_rate=lr,
                                         num_epochs=num_epochs,
                                         )

# Predict on test data
mean, std = predict_variational_gp(model, likelihood, X_test)

nlpd_var_gp = compute_neg_log_like(mean.reshape(-1,1),std.reshape(-1,1),y_test)
# Output results
print("NLPD Variational GP: ", nlpd_var_gp)



