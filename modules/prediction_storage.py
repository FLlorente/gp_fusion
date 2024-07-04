import numpy as np
from .model_training import train_and_predict_single_gp

# Use predict_with_expert and store_predictions_for_experts in model_training.py better!
def store_predictions(splits, X_val, X_test, kappa, lambdaa):
    mu_preds_val = np.zeros((X_val.shape[0], len(splits)))
    std_preds_val = np.zeros((X_val.shape[0], len(splits)))

    mu_preds_test = np.zeros((X_test.shape[0], len(splits)))
    std_preds_test = np.zeros((X_test.shape[0], len(splits)))

    for i, (X_split, y_split) in enumerate(splits):
        test_preds, val_preds = train_and_predict_single_gp(
            X_train=X_split, y_train=y_split, X_test=X_test, X_val=X_val, kappa=kappa, lambdaa=lambdaa
        )

        mu_preds_val[:, i] = val_preds.mean.numpy()
        std_preds_val[:, i] = np.sqrt(val_preds.variance.numpy())

        mu_preds_test[:, i] = test_preds.mean.numpy()
        std_preds_test[:, i] = np.sqrt(test_preds.variance.numpy())

    return mu_preds_val, std_preds_val, mu_preds_test, std_preds_test

