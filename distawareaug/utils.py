"""
Utility functions for data validation, preprocessing, and range clipping.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union
from sklearn.utils import check_array


def validate_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate input data for augmentation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    y : array-like of shape (n_samples,)
        Target labels

    Returns
    -------
    X_validated : array-like of shape (n_samples, n_features)
        Validated feature matrix
    y_validated : array-like of shape (n_samples,)
        Validated target labels

    Raises
    ------
    ValueError
        If data validation fails
    """
    # Validate X
    X = check_array(X, accept_sparse=False, dtype="numeric")

    # Validate y
    y = check_array(y, ensure_2d=False, dtype=None)

    # Check dimensions match
    if X.shape[0] != len(y):
        raise ValueError(
            f"X and y have incompatible shapes: " f"X.shape[0]={X.shape[0]}, y.shape[0]={len(y)}"
        )

    # Check for missing values
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")

    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")

    # Check for infinite values
    if np.any(np.isinf(X)):
        warnings.warn("X contains infinite values", UserWarning)

    # Check minimum number of samples per class
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples_per_class = np.min(class_counts)

    if min_samples_per_class < 2:
        warnings.warn(
            f"Some classes have fewer than 2 samples. " f"Minimum: {min_samples_per_class}",
            UserWarning,
        )

    return X, y


def clip_to_range(
    samples: np.ndarray, reference_data: np.ndarray, method: str = "minmax", margin: float = 0.0
) -> np.ndarray:
    """
    Clip synthetic samples to valid feature ranges.

    Parameters
    ----------
    samples : array-like of shape (n_samples, n_features) or (n_features,)
        Samples to clip
    reference_data : array-like of shape (n_ref, n_features)
        Reference data to determine valid ranges
    method : str, default='minmax'
        Clipping method ('minmax', 'percentile', 'std')
    margin : float, default=0.0
        Additional margin to extend ranges (as fraction of range)

    Returns
    -------
    clipped_samples : array-like
        Samples clipped to valid ranges
    """
    samples = np.atleast_2d(samples)
    reference_data = np.atleast_2d(reference_data)

    if samples.shape[1] != reference_data.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"samples={samples.shape[1]}, reference={reference_data.shape[1]}"
        )

    n_features = samples.shape[1]
    clipped = samples.copy()

    for i in range(n_features):
        ref_feature = reference_data[:, i]

        if method == "minmax":
            min_val = np.min(ref_feature)
            max_val = np.max(ref_feature)
        elif method == "percentile":
            min_val = np.percentile(ref_feature, 5)  # 5th percentile
            max_val = np.percentile(ref_feature, 95)  # 95th percentile
        elif method == "std":
            mean_val = np.mean(ref_feature)
            std_val = np.std(ref_feature)
            min_val = mean_val - 3 * std_val
            max_val = mean_val + 3 * std_val
        else:
            raise ValueError(f"Unknown clipping method: {method}")

        # Add margin
        if margin > 0:
            range_size = max_val - min_val
            margin_size = margin * range_size
            min_val -= margin_size
            max_val += margin_size

        # Clip values
        clipped[:, i] = np.clip(samples[:, i], min_val, max_val)

    # Return in original shape
    if len(clipped) == 1:
        return clipped[0]
    return clipped


def compute_feature_ranges(
    data: np.ndarray, method: str = "minmax"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute valid ranges for each feature.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Data to compute ranges from
    method : str, default='minmax'
        Range computation method ('minmax', 'percentile', 'std')

    Returns
    -------
    min_values : array-like of shape (n_features,)
        Minimum values for each feature
    max_values : array-like of shape (n_features,)
        Maximum values for each feature
    """
    data = np.atleast_2d(data)

    if method == "minmax":
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
    elif method == "percentile":
        min_values = np.percentile(data, 5, axis=0)
        max_values = np.percentile(data, 95, axis=0)
    elif method == "std":
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0)
        min_values = means - 3 * stds
        max_values = means + 3 * stds
    else:
        raise ValueError(f"Unknown range method: {method}")

    return min_values, max_values


def check_class_balance(y: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Check class balance and identify minority/majority classes.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target labels
    threshold : float, default=0.1
        Threshold for considering a class as minority (as fraction of total)

    Returns
    -------
    balance_info : dict
        Dictionary with class balance information
    """
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)

    class_ratios = counts / total_samples

    balance_info = {
        "classes": classes,
        "counts": counts,
        "ratios": class_ratios,
        "majority_class": classes[np.argmax(counts)],
        "minority_classes": classes[class_ratios < threshold],
        "is_balanced": np.std(class_ratios) < 0.1,  # Low standard deviation
        "imbalance_ratio": np.max(counts) / np.min(counts),
    }

    return balance_info


def normalize_features(
    X: np.ndarray, method: str = "standardize", fit_params: Optional[dict] = None
) -> Tuple[np.ndarray, dict]:
    """
    Normalize features for better augmentation performance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix to normalize
    method : str, default='standardize'
        Normalization method ('standardize', 'minmax', 'robust')
    fit_params : dict, default=None
        Pre-computed normalization parameters

    Returns
    -------
    X_normalized : array-like of shape (n_samples, n_features)
        Normalized feature matrix
    norm_params : dict
        Normalization parameters for inverse transform
    """
    X = np.atleast_2d(X)

    if fit_params is None:
        # Compute normalization parameters
        if method == "standardize":
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            norm_params = {"mean": mean, "std": std}
            X_normalized = (X - mean) / std

        elif method == "minmax":
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            norm_params = {"min": min_vals, "range": range_vals}
            X_normalized = (X - min_vals) / range_vals

        elif method == "robust":
            median = np.median(X, axis=0)
            mad = np.median(np.abs(X - median), axis=0)  # Median Absolute Deviation
            mad[mad == 0] = 1  # Avoid division by zero
            norm_params = {"median": median, "mad": mad}
            X_normalized = (X - median) / mad

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    else:
        # Use provided parameters
        if method == "standardize":
            X_normalized = (X - fit_params["mean"]) / fit_params["std"]
        elif method == "minmax":
            X_normalized = (X - fit_params["min"]) / fit_params["range"]
        elif method == "robust":
            X_normalized = (X - fit_params["median"]) / fit_params["mad"]
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        norm_params = fit_params

    return X_normalized, norm_params


def inverse_normalize_features(
    X_normalized: np.ndarray, norm_params: dict, method: str = "standardize"
) -> np.ndarray:
    """
    Inverse normalize features back to original scale.

    Parameters
    ----------
    X_normalized : array-like of shape (n_samples, n_features)
        Normalized feature matrix
    norm_params : dict
        Normalization parameters from normalize_features
    method : str, default='standardize'
        Normalization method used

    Returns
    -------
    X_original : array-like of shape (n_samples, n_features)
        Features in original scale
    """
    X_normalized = np.atleast_2d(X_normalized)

    if method == "standardize":
        X_original = X_normalized * norm_params["std"] + norm_params["mean"]
    elif method == "minmax":
        X_original = X_normalized * norm_params["range"] + norm_params["min"]
    elif method == "robust":
        X_original = X_normalized * norm_params["mad"] + norm_params["median"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return X_original


def detect_categorical_features(
    X: np.ndarray, max_unique_ratio: float = 0.1, max_unique_count: int = 20
) -> np.ndarray:
    """
    Detect which features are likely categorical.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix
    max_unique_ratio : float, default=0.1
        Maximum ratio of unique values to total samples for categorical
    max_unique_count : int, default=20
        Maximum number of unique values for categorical

    Returns
    -------
    is_categorical : array-like of shape (n_features,)
        Boolean array indicating categorical features
    """
    X = np.atleast_2d(X)
    n_samples, n_features = X.shape
    is_categorical = np.zeros(n_features, dtype=bool)

    for i in range(n_features):
        feature = X[:, i]
        n_unique = len(np.unique(feature))
        unique_ratio = n_unique / n_samples

        # Check if feature is likely categorical
        if unique_ratio <= max_unique_ratio or n_unique <= max_unique_count:
            # Additional check: are values integers or discrete?
            if np.all(np.equal(np.mod(feature, 1), 0)):  # All integers
                is_categorical[i] = True

    return is_categorical
