"""
Main oversampling logic for distribution-aware augmentation.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from .distance import DistanceMetrics
from .distribution import DistributionFitter
from .utils import validate_data


class DistAwareAugmentor(BaseEstimator, TransformerMixin):
    """
    Distribution-Aware Data Augmentation for imbalanced datasets.

    This class implements an intelligent oversampling technique that:
    1. Fits distributions to minority class features
    2. Generates new samples from these distributions
    3. Ensures diversity through distance-based filtering
    4. Maintains feature value ranges

    Parameters
    ----------
    sampling_strategy : str or dict, default='auto'
        Strategy for resampling. If 'auto', balance all classes.
        If dict, specifies the number for each class (behavior depends on sampling_mode).
    sampling_mode : str, default='add'
        How to interpret sampling_strategy values:
        - 'add': Add N samples to each class (e.g., {0: 5000, 1: 5000} adds 5000)
        - 'target': Target N total samples for each class (e.g., {0: 5000, 1: 5000}
          results in 5000 total for each)
    diversity_threshold : float, default=0.1
        Minimum distance threshold for accepting new samples
    distribution_method : str, default='kde'
        Method for fitting feature distributions ('kde', 'gaussian', 'uniform')
    distance_metric : str, default='euclidean'
        Distance metric for diversity checking
    random_state : int, default=None
        Random seed for reproducibility
    """

    def __init__(
        self,
        sampling_strategy: str = "auto",
        sampling_mode: str = "add",
        diversity_threshold: float = 0.1,
        distribution_method: str = "kde",
        distance_metric: str = "euclidean",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        self.sampling_strategy = sampling_strategy
        self.sampling_mode = sampling_mode
        self.diversity_threshold = diversity_threshold
        self.distribution_method = distribution_method
        self.distance_metric = distance_metric
        self.random_state = random_state

        # Initialize components
        self.distribution_fitter = DistributionFitter(
            method=distribution_method, random_state=random_state
        )
        self.distance_metrics = DistanceMetrics(metric=distance_metric)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistAwareAugmentor":
        """
        Fit the augmentor to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : DistAwareAugmentor
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        validate_data(X, y)

        self.X_ = X
        self.y_ = y
        self.classes_, self.class_counts_ = np.unique(y, return_counts=True)

        # Create a single RNG for reproducible randomness throughout the object
        self._rng = np.random.RandomState(self.random_state)

        # Determine which classes need augmentation
        self._compute_sampling_strategy()

        # Fit distributions for minority classes
        self.fitted_distributions_ = {}
        for class_label in self.classes_to_augment_:
            class_mask = y == class_label
            class_data = X[class_mask]

            # Create a new DistributionFitter instance for each class
            class_fitter = DistributionFitter(
                method=self.distribution_method, random_state=self.random_state
            )
            self.fitted_distributions_[class_label] = class_fitter.fit(class_data)

        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the augmentor and resample the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        X_resampled : array-like of shape (n_samples_new, n_features)
            Resampled training data
        y_resampled : array-like of shape (n_samples_new,)
            Resampled target labels
        """
        self.fit(X, y)
        return self.resample(X, y)

    def resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset by generating synthetic samples or downsampling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        X_resampled : array-like of shape (n_samples_new, n_features)
            Resampled training data
        y_resampled : array-like of shape (n_samples_new,)
            Resampled target labels
        """
        # Check if any downsampling is needed
        has_downsampling = any(n < 0 for n in self.sampling_strategy_.values())

        if not has_downsampling:
            # Only upsampling: preserve original data order at the beginning
            X_new_list = [X]
            y_new_list = [y]

            for class_label, n_samples in self.sampling_strategy_.items():
                if n_samples > 0:
                    # Generate synthetic samples for this class
                    synthetic_X = self._generate_samples(class_label, n_samples)
                    synthetic_y = np.full(n_samples, class_label)

                    X_new_list.append(synthetic_X)
                    y_new_list.append(synthetic_y)

            X_resampled = np.vstack(X_new_list)
            y_resampled = np.hstack(y_new_list)
        else:
            # Has downsampling: process each class separately
            X_new_list = []
            y_new_list = []

            for class_label in self.classes_:
                class_mask = y == class_label
                class_X = X[class_mask]
                class_y = y[class_mask]

                n_samples_needed = self.sampling_strategy_.get(class_label, 0)

                if n_samples_needed > 0:
                    # Upsample: add original + synthetic samples
                    synthetic_X = self._generate_samples(class_label, n_samples_needed)
                    synthetic_y = np.full(n_samples_needed, class_label)

                    X_new_list.append(class_X)
                    y_new_list.append(class_y)
                    X_new_list.append(synthetic_X)
                    y_new_list.append(synthetic_y)

                elif n_samples_needed < 0:
                    # Downsample: randomly select samples to keep
                    n_samples_to_keep = (
                        len(class_X) + n_samples_needed
                    )  # n_samples_needed is negative
                    indices = self._rng.choice(len(class_X), size=n_samples_to_keep, replace=False)
                    X_new_list.append(class_X[indices])
                    y_new_list.append(class_y[indices])

                else:
                    # No change needed, keep original samples
                    X_new_list.append(class_X)
                    y_new_list.append(class_y)

            X_resampled = np.vstack(X_new_list)
            y_resampled = np.hstack(y_new_list)

        return X_resampled, y_resampled

    def _compute_sampling_strategy(self):
        """Compute how many samples to generate for each class."""
        # Validate sampling_mode
        if self.sampling_mode not in ["add", "target"]:
            raise ValueError(f"sampling_mode must be 'add' or 'target', got '{self.sampling_mode}'")

        if self.sampling_strategy == "auto":
            # Balance all classes to majority class count
            max_count = np.max(self.class_counts_)
            self.sampling_strategy_ = {}

            for class_label, count in zip(self.classes_, self.class_counts_):
                self.sampling_strategy_[class_label] = max_count - count

        else:
            # Handle custom sampling strategies based on sampling_mode
            self.sampling_strategy_ = {}

            for class_label, count in zip(self.classes_, self.class_counts_):
                if class_label in self.sampling_strategy:
                    target_value = self.sampling_strategy[class_label]

                    if self.sampling_mode == "add":
                        # Add mode: add N samples to existing count
                        self.sampling_strategy_[class_label] = target_value
                    else:  # sampling_mode == 'target'
                        # Target mode: target N total samples
                        # Calculate difference (positive = upsample, negative = downsample)
                        self.sampling_strategy_[class_label] = target_value - count
                else:
                    # Class not specified in strategy, don't modify
                    self.sampling_strategy_[class_label] = 0

        # Identify classes that need augmentation (upsampling only)
        self.classes_to_augment_ = [
            cls for cls, n_samples in self.sampling_strategy_.items() if n_samples > 0
        ]

    def _generate_samples(self, class_label: int, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a specific class using batch generation."""
        class_mask = self.y_ == class_label
        class_data = self.X_[class_mask]

        # Get the fitted distribution fitter for this class
        distributions = self.fitted_distributions_[class_label]

        # Initialize NearestNeighbors for efficient distance checking
        # Use all available cores for neighbor queries where supported
        nn_original = NearestNeighbors(n_neighbors=1, metric=self.distance_metric, n_jobs=-1)
        nn_original.fit(class_data)

        synthetic_samples = []
        # Adaptive batch size: larger upper bound to reduce Python-level loop overhead
        batch_size = min(2000, max(500, n_samples // 2))  # Adaptive batch size
        attempts = 0
        max_attempts = (n_samples * 10) // batch_size + 1  # Adjust for batch generation

        while len(synthetic_samples) < n_samples and attempts < max_attempts:
            # Generate a batch of candidates
            batch = self._generate_candidate_batch(distributions, class_data, batch_size)

            # Filter batch based on diversity constraints
            diverse_samples = self._filter_diverse_batch(
                batch, nn_original, synthetic_samples, class_data
            )

            synthetic_samples.extend(diverse_samples)
            attempts += 1

        # Convert to array and trim to exact size needed
        synthetic_array = np.array(synthetic_samples[:n_samples])

        # If we couldn't generate enough diverse samples, fill with additional samples
        if len(synthetic_array) < n_samples:
            remaining = n_samples - len(synthetic_array)
            additional_batch = self._generate_candidate_batch(distributions, class_data, remaining)
            synthetic_array = np.vstack([synthetic_array, additional_batch[:remaining]])

        return synthetic_array

    def _generate_candidate_batch(
        self, distributions, class_data: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Generate a batch of candidate samples from fitted distributions."""
        n_features = len(distributions.distributions)
        candidates = np.zeros((batch_size, n_features))

        # Generate samples for each feature
        # (distribution.sample should return shape (batch_size, 1) or similar)
        for i, dist in distributions.distributions.items():
            candidates[:, i] = dist.sample(batch_size).flatten()

        # Vectorized clipping: compute per-feature mins and maxs from class_data
        mins = class_data.min(axis=0)
        maxs = class_data.max(axis=0)
        candidates = np.clip(candidates, mins, maxs)

        return candidates

    def _filter_diverse_batch(
        self,
        candidates: np.ndarray,
        nn_original: NearestNeighbors,
        synthetic_samples: list,
        class_data: np.ndarray,
    ) -> list:
        """Filter a batch of candidates based on diversity constraints."""
        diverse = []

        # Check distance to original samples for all candidates at once
        distances_to_original, _ = nn_original.kneighbors(candidates)
        distances_to_original = distances_to_original.flatten()

        # Pre-filter candidates that are too close to original data
        valid_mask = distances_to_original >= self.diversity_threshold
        valid_candidates = candidates[valid_mask]

        if len(valid_candidates) == 0:
            return diverse

        # If synthetic samples exist, check diversity against a subsample
        if synthetic_samples:
            synthetic_array = np.array(synthetic_samples)

            # PERFORMANCE FIX: Only check against a random subsample of synthetic samples
            # This prevents O(n²) growth as synthetic_samples accumulates
            max_check_size = 200  # Check against at most 200 synthetic samples

            if len(synthetic_array) > max_check_size:
                # Randomly subsample synthetic samples for diversity check
                indices = self._rng.choice(len(synthetic_array), size=max_check_size, replace=False)
                synthetic_check = synthetic_array[indices]
            else:
                synthetic_check = synthetic_array

            # Compute minimal distance from each valid candidate to synthetic subsample
            dists = pairwise_distances(
                valid_candidates, synthetic_check, metric=self.distance_metric
            )
            min_dists = dists.min(axis=1)

            # Filter candidates that are sufficiently far from synthetic samples
            final_mask = min_dists >= self.diversity_threshold
            diverse = valid_candidates[final_mask].tolist()
        else:
            diverse = valid_candidates.tolist()

        return diverse
