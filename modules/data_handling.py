import numpy as np
from sklearn.model_selection import train_test_split
from uci_datasets import Dataset

def load_and_normalize_data(dataset_name, split=0):
    data = Dataset(dataset_name);
    X_train, y_train, X_test, y_test = data.get_split(split=split)
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    X_train_max = X_train.max(0)
    X_train_min = X_train.min(0)
    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

    return X_train, y_train, X_test, y_test

def split_dataset(X, Y, n_splits, split_size, with_replacement=True):
    np.random.seed(0)
    n_samples = X.shape[0]

    if with_replacement:
        splits = [(X[indices], Y[indices]) for indices in [np.random.choice(n_samples, split_size, replace=True) for _ in range(n_splits)]]
    else:
        indices = np.random.permutation(n_samples)
        splits = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if (i + 1) * split_size < n_samples else n_samples
            splits.append((X[indices[start_idx:end_idx]], Y[indices[start_idx:end_idx]]))
        # Ensure all remaining samples are included in the final split
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

