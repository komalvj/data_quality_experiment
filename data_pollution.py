import numpy as np
import pandas as pd

def add_noise(X, noise_level=0.1):
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def introduce_inconsistent_representation(X):
    X_inconsistent = X.copy()
    X_inconsistent['inconsistent_feature'] = np.where(np.random.rand(len(X)) > 0.5, 'NYC', 'New York')
    return X_inconsistent

def reduce_feature_accuracy(X, perturbation_level=0.1):
    np.random.seed(42)
    perturbation = np.random.normal(0, perturbation_level, X.shape)
    return X + perturbation

def reduce_target_accuracy(y, mislabel_rate=0.1):
    np.random.seed(42)
    y_inaccurate = y.copy()
    mislabel_indices = np.random.choice(len(y), int(mislabel_rate * len(y)), replace=False)
    y_inaccurate[mislabel_indices] = np.random.choice(np.unique(y), len(mislabel_indices))
    return y_inaccurate

def reduce_uniqueness(X, y, duplicate_fraction=0.1):
    np.random.seed(42)
    num_duplicates = int(duplicate_fraction * len(X))
    duplicates = X.iloc[:num_duplicates]
    X_duplicate = pd.concat([X, duplicates], ignore_index=True)
    y_duplicate = pd.concat([y, y.iloc[:num_duplicates]], ignore_index=True)
    return X_duplicate, y_duplicate
