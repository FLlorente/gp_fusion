import numpy as np
from sklearn.model_selection import train_test_split
from uci_datasets import Dataset

def load_and_normalize_data(dataset_name, split=0, normalize_y = False, normalize_x_method="max-min"):
    data = Dataset(dataset_name,print_stats=False)

    X_train, y_train, X_test, y_test = data.get_split(split=split)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # remove columns that are zero
    if dataset_name=="autos":  
        X_train = np.delete(X_train,[8],axis=1)
        X_test = np.delete(X_test,[8],axis=1)

    if dataset_name == "sml":
        X_train = np.delete(X_train,[2, 20, 21, 22],axis=1)
        X_test = np.delete(X_test,[2, 20, 21, 22],axis=1)
   
    if normalize_x_method == "max-min":
        X_train_max = X_train.max(0)
        X_train_min = X_train.min(0)
        X_train = (X_train - X_train_min) / ((X_train_max - X_train_min) + 1e-6)
        X_test = (X_test - X_train_min) / ((X_train_max - X_train_min) + 1e-6)
    if normalize_x_method == "z-score":
        X_train_mean = X_train.mean(0)
        X_train_std = np.std(X_train,axis=0)
        X_train = (X_train - X_train_mean) / (X_train_std + 1e-6)
        X_test = (X_test - X_train_mean) / (X_train_std + 1e-6)    

    if normalize_y:
        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train) + 1e-6

        y_train  = (y_train - y_train_mean)/y_train_std
        y_test  = (y_test - y_train_mean)/y_train_std

    return X_train, y_train, X_test, y_test

def split_dataset(X, Y, n_splits=None, split_size=None, with_replacement=True):
    np.random.seed(0)
    n_samples = X.shape[0]

    if with_replacement:
        splits = [(X[indices], Y[indices]) for indices in [np.random.choice(n_samples, split_size, replace=True) for _ in range(n_splits)]]
    else:
        # if one of them is missing
        if n_splits is None:
            n_splits = np.ceil(n_samples/split_size).astype(int) 
        if split_size is None:
            split_size = np.ceil(n_samples/n_splits).astype(int) 

        indices = np.random.permutation(n_samples)
        splits = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if (i + 1) * split_size < n_samples else n_samples
            splits.append((X[indices[start_idx:end_idx]], Y[indices[start_idx:end_idx]]))
        # Ensure all remaining samples are included in the final split (I want to avoid this, I prefer to have all (but one) experts with more data than viceversa...)
        if n_splits * split_size < n_samples:
            splits[-1] = (np.concatenate((splits[-1][0], X[indices[n_splits * split_size:]]), axis=0),
                          np.concatenate((splits[-1][1], Y[indices[n_splits * split_size:]]), axis=0))

    return splits


def create_validation_set(splits, n_points_per_split):
    np.random.seed(0)
    X_val = []
    y_val = []

    for X_split, y_split in splits:
        selected_indices = np.random.choice(len(X_split), n_points_per_split, replace=False)
        X_val.append(X_split[selected_indices])
        y_val.append(y_split[selected_indices])

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    return X_val, y_val

