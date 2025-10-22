"""
Tests for distance metrics and diversity checking functionality.
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distawareaug.distance import DistanceMetrics


class TestDistanceMetrics(unittest.TestCase):
    """Test cases for DistanceMetrics class."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create test data
        self.X_small = np.array([[0, 0], [1, 1], [2, 2]])
        self.Y_small = np.array([[0.5, 0.5], [1.5, 1.5]])

        # Create larger test dataset
        self.X_large, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        self.Y_large, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=123)

    def test_initialization_default(self):
        """Test default initialization."""
        dm = DistanceMetrics()
        self.assertEqual(dm.metric, "euclidean")
        self.assertEqual(dm.kwargs, {})

    def test_initialization_custom(self):
        """Test custom initialization."""
        dm = DistanceMetrics(metric="manhattan")
        self.assertEqual(dm.metric, "manhattan")

        dm_minkowski = DistanceMetrics(metric="minkowski", p=3)
        self.assertEqual(dm_minkowski.metric, "minkowski")
        self.assertEqual(dm_minkowski.kwargs["p"], 3)

    def test_invalid_metric(self):
        """Test error handling for invalid metrics."""
        with self.assertRaises(ValueError):
            DistanceMetrics(metric="invalid_metric")

    def test_compute_distances_euclidean(self):
        """Test Euclidean distance computation."""
        dm = DistanceMetrics(metric="euclidean")
        distances = dm.compute_distances(self.X_small, self.Y_small)

        # Check shape
        self.assertEqual(distances.shape, (3, 2))

        # Check specific values
        expected = euclidean_distances(self.X_small, self.Y_small)
        np.testing.assert_array_almost_equal(distances, expected, decimal=10)

    def test_compute_distances_manhattan(self):
        """Test Manhattan distance computation."""
        dm = DistanceMetrics(metric="manhattan")
        distances = dm.compute_distances(self.X_small, self.Y_small)

        # Check shape
        self.assertEqual(distances.shape, (3, 2))

        # Check specific values
        expected = manhattan_distances(self.X_small, self.Y_small)
        np.testing.assert_array_almost_equal(distances, expected, decimal=10)

    def test_compute_distances_self(self):
        """Test distance computation with Y=None (self-distances)."""
        dm = DistanceMetrics(metric="euclidean")
        distances = dm.compute_distances(self.X_small)

        # Check shape
        self.assertEqual(distances.shape, (3, 3))

        # Diagonal should be zero
        np.testing.assert_array_almost_equal(np.diag(distances), [0, 0, 0])

        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_nearest_neighbor_distances(self):
        """Test nearest neighbor distance computation."""
        dm = DistanceMetrics(metric="euclidean")

        # Test with Y=None (nearest neighbor within X)
        nn_distances = dm.nearest_neighbor_distances(self.X_small)
        self.assertEqual(len(nn_distances), 3)

        # All distances should be positive (no self-distances)
        self.assertTrue(np.all(nn_distances > 0))

        # Test with separate Y
        nn_distances_Y = dm.nearest_neighbor_distances(self.X_small, self.Y_small)
        self.assertEqual(len(nn_distances_Y), 3)

    def test_k_nearest_distances(self):
        """Test k-nearest neighbor distances."""
        dm = DistanceMetrics(metric="euclidean")

        # Test with k=2
        k = 2
        knn_distances = dm.k_nearest_distances(self.X_large, k=k)

        # Check shape
        self.assertEqual(knn_distances.shape, (len(self.X_large), k))

        # Distances should be sorted (ascending)
        for i in range(len(self.X_large)):
            for j in range(k - 1):
                self.assertLessEqual(knn_distances[i, j], knn_distances[i, j + 1])

    def test_diversity_score(self):
        """Test diversity score computation."""
        dm = DistanceMetrics(metric="euclidean")

        # Test internal diversity only
        diversity = dm.diversity_score(self.X_small)
        self.assertIsInstance(diversity, float)
        self.assertGreaterEqual(diversity, 0)

        # Test with reference samples
        diversity_ext = dm.diversity_score(self.X_small, self.Y_small)
        self.assertIsInstance(diversity_ext, float)
        self.assertGreaterEqual(diversity_ext, 0)

        # Edge case: single sample
        single_sample = self.X_small[:1]
        diversity_single = dm.diversity_score(single_sample)
        self.assertEqual(diversity_single, 0.0)

    def test_is_diverse_enough(self):
        """Test diversity checking for individual samples."""
        dm = DistanceMetrics(metric="euclidean")

        # Test with different thresholds
        candidate = np.array([0.1, 0.1])
        existing = self.X_small

        # Very low threshold - should be diverse enough
        self.assertTrue(dm.is_diverse_enough(candidate, existing, threshold=0.01))

        # Very high threshold - should not be diverse enough
        self.assertFalse(dm.is_diverse_enough(candidate, existing, threshold=10.0))

        # Empty existing samples - should always be diverse
        self.assertTrue(dm.is_diverse_enough(candidate, np.array([]).reshape(0, 2), threshold=1.0))

    def test_filter_diverse_samples(self):
        """Test filtering of diverse samples."""
        dm = DistanceMetrics(metric="euclidean")

        # Create samples with some that are too close
        samples = np.array(
            [
                [0, 0],  # Keep
                [0.01, 0.01],  # Too close to first, filter out
                [5, 5],  # Keep
                [5.01, 5.01],  # Too close to third, filter out
                [10, 10],  # Keep
            ]
        )

        diverse_samples = dm.filter_diverse_samples(samples, threshold=0.1)

        # Should keep fewer samples
        self.assertLess(len(diverse_samples), len(samples))
        self.assertGreater(len(diverse_samples), 0)

        # Test with max_samples limit
        diverse_samples_limited = dm.filter_diverse_samples(samples, threshold=0.01, max_samples=2)
        self.assertLessEqual(len(diverse_samples_limited), 2)

    def test_filter_diverse_samples_with_reference(self):
        """Test filtering diverse samples with reference data."""
        dm = DistanceMetrics(metric="euclidean")

        # Reference samples
        reference = np.array([[0, 0], [1, 1]])

        # Candidate samples (some close to reference)
        candidates = np.array(
            [
                [0.01, 0.01],  # Too close to reference
                [5, 5],  # Far from reference, should keep
                [1.01, 1.01],  # Too close to reference
                [10, 10],  # Far from reference, should keep
            ]
        )

        diverse_samples = dm.filter_diverse_samples(candidates, reference=reference, threshold=0.5)

        # Should filter out samples close to reference
        self.assertLess(len(diverse_samples), len(candidates))

    def test_adaptive_threshold(self):
        """Test adaptive threshold computation."""
        dm = DistanceMetrics(metric="euclidean")

        # Test with clustered data
        clustered_data = np.array(
            [
                [0, 0],
                [0.1, 0.1],
                [0.2, 0.2],  # Tight cluster
                [10, 10],
                [10.1, 10.1],
                [10.2, 10.2],  # Another tight cluster
            ]
        )

        threshold = dm.adaptive_threshold(clustered_data)
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)

        # Test with different target ratios
        threshold_low = dm.adaptive_threshold(clustered_data, target_ratio=0.2)
        threshold_high = dm.adaptive_threshold(clustered_data, target_ratio=0.8)

        # Higher ratio should give higher threshold
        self.assertGreater(threshold_high, threshold_low)

        # Edge case: single sample
        single_sample = clustered_data[:1]
        threshold_single = dm.adaptive_threshold(single_sample)
        self.assertEqual(threshold_single, 0.1)  # Default fallback

    def test_different_distance_metrics_consistency(self):
        """Test that different metrics produce reasonable results."""
        X = self.X_large[:10]  # Small subset for testing

        metrics_to_test = ["euclidean", "manhattan", "cosine"]

        for metric in metrics_to_test:
            with self.subTest(metric=metric):
                dm = DistanceMetrics(metric=metric)

                # Test basic distance computation
                distances = dm.compute_distances(X)
                self.assertEqual(distances.shape, (10, 10))

                # Diagonal should be zero (or very close for cosine)
                diagonal = np.diag(distances)
                if metric == "cosine":
                    np.testing.assert_array_almost_equal(diagonal, np.zeros(10), decimal=10)
                else:
                    np.testing.assert_array_equal(diagonal, np.zeros(10))

                # Test diversity score
                diversity = dm.diversity_score(X)
                self.assertIsInstance(diversity, float)
                self.assertGreaterEqual(diversity, 0)

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        dm = DistanceMetrics(metric="euclidean")

        # This is more of a smoke test to ensure no errors with larger data
        distances = dm.compute_distances(self.X_large, self.Y_large)
        self.assertEqual(distances.shape, (len(self.X_large), len(self.Y_large)))

        diversity = dm.diversity_score(self.X_large)
        self.assertIsInstance(diversity, float)

    def test_edge_cases(self):
        """Test various edge cases."""
        dm = DistanceMetrics(metric="euclidean")

        # Empty arrays
        empty = np.array([]).reshape(0, 2)
        diverse_empty = dm.filter_diverse_samples(empty)
        self.assertEqual(len(diverse_empty), 0)

        # Single point
        single_point = np.array([[1, 1]])
        distances_single = dm.compute_distances(single_point)
        self.assertEqual(distances_single.shape, (1, 1))
        self.assertEqual(distances_single[0, 0], 0)

        # Identical points
        identical_points = np.array([[1, 1], [1, 1], [1, 1]])
        nn_dist = dm.nearest_neighbor_distances(identical_points)
        np.testing.assert_array_almost_equal(nn_dist, [0, 0, 0])


class TestDistanceMetricsValidation(unittest.TestCase):
    """Test validation and error handling in DistanceMetrics."""

    def test_input_validation(self):
        """Test input validation for distance computation."""
        dm = DistanceMetrics()

        # Test with mismatched dimensions
        X = np.array([[1, 2]])
        Y = np.array([[1, 2, 3]])  # Different number of features

        # This should raise an error during distance computation
        with self.assertRaises((ValueError, Exception)):
            dm.compute_distances(X, Y)

    def test_minkowski_parameter(self):
        """Test Minkowski distance with different p values."""
        dm_l1 = DistanceMetrics(metric="minkowski", p=1)  # Should be same as Manhattan
        dm_l2 = DistanceMetrics(metric="minkowski", p=2)  # Should be same as Euclidean

        X = np.array([[0, 0], [1, 1]])

        distances_l1 = dm_l1.compute_distances(X)
        distances_l2 = dm_l2.compute_distances(X)

        # L1 and L2 should give different results (except for diagonal)
        self.assertNotEqual(distances_l1[0, 1], distances_l2[0, 1])

        # But diagonals should be zero
        np.testing.assert_array_equal(np.diag(distances_l1), [0, 0])
        np.testing.assert_array_equal(np.diag(distances_l2), [0, 0])


if __name__ == "__main__":
    unittest.main()
