"""
Distance metrics and diversity checking for synthetic sample generation.
"""

import numpy as np
from typing import Union, Optional, Callable
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import minkowski


class DistanceMetrics:
    """
    Distance computation and diversity checking utilities.

    Parameters
    ----------
    metric : str or callable, default='euclidean'
        Distance metric to use. Supported: 'euclidean', 'manhattan',
        'cosine', 'minkowski', or custom callable
    **kwargs : dict
        Additional parameters for distance metrics (e.g., p for Minkowski)
    """

    def __init__(self, metric: Union[str, Callable] = "euclidean", **kwargs):
        self.metric = metric
        self.kwargs = kwargs

        # Validate metric
        self._validate_metric()

    def _validate_metric(self):
        """Validate the distance metric."""
        if isinstance(self.metric, str):
            supported_metrics = {
                "euclidean",
                "manhattan",
                "cosine",
                "minkowski",
                "chebyshev",
                "hamming",
                "jaccard",
            }
            if self.metric not in supported_metrics:
                raise ValueError(f"Unsupported metric: {self.metric}")

    def compute_distances(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances between samples.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            First set of samples
        Y : array-like of shape (n_samples_Y, n_features), default=None
            Second set of samples. If None, compute distances within X

        Returns
        -------
        distances : array-like of shape (n_samples_X, n_samples_Y)
            Pairwise distances
        """
        if Y is None:
            Y = X

        # Use optimized functions for common metrics
        if self.metric == "euclidean":
            return euclidean_distances(X, Y)
        elif self.metric == "manhattan":
            return manhattan_distances(X, Y)
        elif self.metric == "cosine":
            return cosine_distances(X, Y)
        else:
            # Use sklearn's general pairwise_distances
            return pairwise_distances(X, Y, metric=self.metric, **self.kwargs)

    def nearest_neighbor_distances(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute distances to nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Query samples
        Y : array-like of shape (n_samples_Y, n_features), default=None
            Reference samples. If None, use X (excluding self-distances)

        Returns
        -------
        nn_distances : array-like of shape (n_samples_X,)
            Distance to nearest neighbor for each sample in X
        """
        distances = self.compute_distances(X, Y)

        if Y is None:
            # Exclude self-distances by setting diagonal to infinity
            np.fill_diagonal(distances, np.inf)

        return np.min(distances, axis=1)

    def k_nearest_distances(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None, k: int = 5
    ) -> np.ndarray:
        """
        Compute distances to k nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Query samples
        Y : array-like of shape (n_samples_Y, n_features), default=None
            Reference samples. If None, use X
        k : int, default=5
            Number of nearest neighbors

        Returns
        -------
        knn_distances : array-like of shape (n_samples_X, k)
            Distances to k nearest neighbors for each sample in X
        """
        distances = self.compute_distances(X, Y)

        if Y is None:
            # Exclude self-distances
            np.fill_diagonal(distances, np.inf)

        # Sort distances and take k smallest
        sorted_distances = np.sort(distances, axis=1)
        return sorted_distances[:, :k]

    def diversity_score(self, samples: np.ndarray, reference: Optional[np.ndarray] = None) -> float:
        """
        Compute diversity score for a set of samples.

        Higher scores indicate more diverse samples.

        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Samples to evaluate
        reference : array-like of shape (n_ref, n_features), default=None
            Reference samples for external diversity. If None, compute
            internal diversity only

        Returns
        -------
        diversity : float
            Diversity score (higher = more diverse)
        """
        if len(samples) < 2:
            return 0.0

        # Internal diversity: average pairwise distance within samples
        internal_distances = self.compute_distances(samples)
        # Exclude diagonal (self-distances)
        mask = ~np.eye(len(samples), dtype=bool)
        internal_diversity = np.mean(internal_distances[mask])

        if reference is None:
            return internal_diversity

        # External diversity: average distance to reference samples
        external_distances = self.compute_distances(samples, reference)
        external_diversity = np.mean(external_distances)

        # Combine internal and external diversity
        return 0.5 * internal_diversity + 0.5 * external_diversity

    def is_diverse_enough(
        self, candidate: np.ndarray, existing_samples: np.ndarray, threshold: float = 0.1
    ) -> bool:
        """
        Check if a candidate sample is diverse enough from existing samples.

        Parameters
        ----------
        candidate : array-like of shape (n_features,)
            Candidate sample to check
        existing_samples : array-like of shape (n_samples, n_features)
            Existing samples to compare against
        threshold : float, default=0.1
            Minimum distance threshold for diversity

        Returns
        -------
        is_diverse : bool
            True if candidate is diverse enough, False otherwise
        """
        if len(existing_samples) == 0:
            return True

        candidate = candidate.reshape(1, -1)
        distances = self.compute_distances(candidate, existing_samples)
        min_distance = np.min(distances)

        return min_distance >= threshold

    def filter_diverse_samples(
        self,
        samples: np.ndarray,
        reference: Optional[np.ndarray] = None,
        threshold: float = 0.1,
        max_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        Filter samples to keep only diverse ones.

        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Candidate samples to filter
        reference : array-like of shape (n_ref, n_features), default=None
            Reference samples for diversity checking
        threshold : float, default=0.1
            Minimum distance threshold for diversity
        max_samples : int, default=None
            Maximum number of samples to return

        Returns
        -------
        diverse_samples : array-like of shape (n_diverse, n_features)
            Filtered diverse samples
        """
        if len(samples) == 0:
            return samples

        diverse_indices = []

        for i, sample in enumerate(samples):
            # Check diversity against reference samples
            if reference is not None:
                if not self.is_diverse_enough(sample, reference, threshold):
                    continue

            # Check diversity against already selected diverse samples
            if diverse_indices:
                selected_samples = samples[diverse_indices]
                if not self.is_diverse_enough(sample, selected_samples, threshold):
                    continue

            diverse_indices.append(i)

            # Stop if we have enough samples
            if max_samples and len(diverse_indices) >= max_samples:
                break

        return (
            samples[diverse_indices]
            if diverse_indices
            else np.array([]).reshape(0, samples.shape[1])
        )

    def adaptive_threshold(self, samples: np.ndarray, target_ratio: float = 0.5) -> float:
        """
        Compute adaptive diversity threshold based on sample distribution.

        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Samples to analyze
        target_ratio : float, default=0.5
            Target ratio of distances to use as threshold

        Returns
        -------
        threshold : float
            Adaptive diversity threshold
        """
        if len(samples) < 2:
            return 0.1  # Default threshold

        distances = self.compute_distances(samples)
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        all_distances = distances[mask]

        # Use percentile as threshold
        threshold = np.percentile(all_distances, target_ratio * 100)

        return max(threshold, 1e-6)  # Ensure non-zero threshold
