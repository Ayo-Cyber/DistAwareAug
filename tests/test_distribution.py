"""
Tests for distribution fitting and sampling functionality.
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distawareaug.distribution import (
    DistributionFitter,
    KDEDistribution,
    GaussianDistribution,
    UniformDistribution,
)


class TestBaseDistributions(unittest.TestCase):
    """Test cases for base distribution classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test data
        np.random.seed(42)
        self.data_1d = np.random.normal(0, 1, 100)
        self.data_2d = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)

    def test_kde_distribution_1d(self):
        """Test KDE distribution with 1D data."""
        kde = KDEDistribution()
        kde.fit(self.data_1d)

        # Test sampling
        samples = kde.sample(50)
        self.assertEqual(len(samples), 50)
        self.assertTrue(isinstance(samples, np.ndarray))

    def test_kde_distribution_2d(self):
        """Test KDE distribution with 2D data."""
        kde = KDEDistribution()
        kde.fit(self.data_2d)

        # Test sampling
        samples = kde.sample(50)
        self.assertEqual(samples.shape, (50, 2))

    def test_gaussian_distribution_1d(self):
        """Test Gaussian distribution with 1D data."""
        gaussian = GaussianDistribution()
        gaussian.fit(self.data_1d)

        # Check fitted parameters
        self.assertAlmostEqual(gaussian.mean_[0], np.mean(self.data_1d), places=2)
        self.assertAlmostEqual(gaussian.cov_[0, 0], np.var(self.data_1d), places=2)

        # Test sampling
        samples = gaussian.sample(50)
        self.assertEqual(len(samples), 50)

    def test_gaussian_distribution_2d(self):
        """Test Gaussian distribution with 2D data."""
        gaussian = GaussianDistribution()
        gaussian.fit(self.data_2d)

        # Check fitted parameters
        np.testing.assert_array_almost_equal(
            gaussian.mean_, np.mean(self.data_2d, axis=0), decimal=2
        )

        # Test sampling
        samples = gaussian.sample(50)
        self.assertEqual(samples.shape, (50, 2))

    def test_uniform_distribution_1d(self):
        """Test Uniform distribution with 1D data."""
        uniform = UniformDistribution()
        uniform.fit(self.data_1d)

        # Check bounds
        expected_min = np.min(self.data_1d)
        expected_max = np.max(self.data_1d)
        self.assertAlmostEqual(uniform.bounds_[0, 0], expected_min, places=5)
        self.assertAlmostEqual(uniform.bounds_[0, 1], expected_max, places=5)

        # Test sampling
        samples = uniform.sample(50)
        self.assertEqual(len(samples), 50)
        self.assertTrue(np.all(samples >= expected_min))
        self.assertTrue(np.all(samples <= expected_max))

    def test_uniform_distribution_2d(self):
        """Test Uniform distribution with 2D data."""
        uniform = UniformDistribution()
        uniform.fit(self.data_2d)

        # Test sampling
        samples = uniform.sample(50)
        self.assertEqual(samples.shape, (50, 2))

        # Check bounds are respected
        for i in range(2):
            expected_min = np.min(self.data_2d[:, i])
            expected_max = np.max(self.data_2d[:, i])
            self.assertTrue(np.all(samples[:, i] >= expected_min))
            self.assertTrue(np.all(samples[:, i] <= expected_max))


class TestDistributionFitter(unittest.TestCase):
    """Test cases for DistributionFitter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test data with known structure
        np.random.seed(42)
        self.X_1d = np.random.normal(5, 2, 100).reshape(-1, 1)
        self.X_2d = np.random.multivariate_normal([1, 2], [[4, 1], [1, 3]], 100)

    def test_initialization(self):
        """Test DistributionFitter initialization."""
        fitter = DistributionFitter()
        self.assertEqual(fitter.method, "kde")
        self.assertIsNone(fitter.random_state)

        fitter_custom = DistributionFitter(method="gaussian", random_state=42)
        self.assertEqual(fitter_custom.method, "gaussian")
        self.assertEqual(fitter_custom.random_state, 42)

    def test_fit_kde(self):
        """Test fitting with KDE method."""
        fitter = DistributionFitter(method="kde", random_state=42)
        distributions = fitter.fit(self.X_2d)

        self.assertEqual(len(distributions), 2)  # One per feature
        self.assertIn(0, distributions)
        self.assertIn(1, distributions)
        self.assertIsInstance(distributions[0], KDEDistribution)

    def test_fit_gaussian(self):
        """Test fitting with Gaussian method."""
        fitter = DistributionFitter(method="gaussian", random_state=42)
        distributions = fitter.fit(self.X_2d)

        self.assertEqual(len(distributions), 2)
        self.assertIsInstance(distributions[0], GaussianDistribution)

    def test_fit_uniform(self):
        """Test fitting with Uniform method."""
        fitter = DistributionFitter(method="uniform", random_state=42)
        distributions = fitter.fit(self.X_2d)

        self.assertEqual(len(distributions), 2)
        self.assertIsInstance(distributions[0], UniformDistribution)

    def test_sample_after_fit(self):
        """Test sampling after fitting."""
        fitter = DistributionFitter(method="kde", random_state=42)
        fitter.fit(self.X_2d)

        samples = fitter.sample(50)
        self.assertEqual(samples.shape, (50, 2))

    def test_fit_sample_combined(self):
        """Test fit_sample method."""
        fitter = DistributionFitter(method="gaussian", random_state=42)
        samples = fitter.fit_sample(self.X_2d, 30)

        self.assertEqual(samples.shape, (30, 2))

    def test_1d_data_handling(self):
        """Test handling of 1D data."""
        fitter = DistributionFitter(method="gaussian", random_state=42)

        # Test with 1D array
        data_1d_flat = self.X_1d.flatten()
        distributions = fitter.fit(data_1d_flat)

        self.assertEqual(len(distributions), 1)
        samples = fitter.sample(20)
        self.assertEqual(samples.shape, (20, 1))

    def test_invalid_method(self):
        """Test error handling for invalid methods."""
        fitter = DistributionFitter(method="invalid_method")

        with self.assertRaises(ValueError):
            fitter.fit(self.X_2d)

    def test_sample_without_fit(self):
        """Test error when sampling without fitting."""
        fitter = DistributionFitter()

        with self.assertRaises(ValueError):
            fitter.sample(10)

    def test_reproducibility_with_random_state(self):
        """Test that random_state ensures reproducible results."""
        fitter1 = DistributionFitter(method="gaussian", random_state=42)
        fitter2 = DistributionFitter(method="gaussian", random_state=42)

        samples1 = fitter1.fit_sample(self.X_2d, 50)
        samples2 = fitter2.fit_sample(self.X_2d, 50)

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_sample_sizes(self):
        """Test sampling different numbers of samples."""
        fitter = DistributionFitter(method="kde", random_state=42)
        fitter.fit(self.X_2d)

        for n_samples in [1, 10, 100]:
            with self.subTest(n_samples=n_samples):
                samples = fitter.sample(n_samples)
                self.assertEqual(samples.shape[0], n_samples)
                self.assertEqual(samples.shape[1], 2)


class TestDistributionQuality(unittest.TestCase):
    """Test the quality and correctness of fitted distributions."""

    def setUp(self):
        """Set up test data with known properties."""
        np.random.seed(42)

        # Create data with known mean and variance
        self.normal_data = np.random.normal(10, 3, 1000)
        self.uniform_data = np.random.uniform(0, 20, 1000)

    def test_gaussian_fitting_accuracy(self):
        """Test that Gaussian distribution captures data statistics accurately."""
        gaussian = GaussianDistribution()
        gaussian.fit(self.normal_data)

        # Check that fitted parameters are close to true parameters
        true_mean = 10
        true_std = 3

        self.assertAlmostEqual(gaussian.mean_[0], true_mean, delta=0.5)
        self.assertAlmostEqual(np.sqrt(gaussian.cov_[0, 0]), true_std, delta=0.5)

    def test_uniform_fitting_accuracy(self):
        """Test that Uniform distribution captures data range accurately."""
        uniform = UniformDistribution()
        uniform.fit(self.uniform_data)

        # Check bounds
        true_min, true_max = 0, 20
        fitted_min = uniform.bounds_[0, 0]
        fitted_max = uniform.bounds_[0, 1]

        self.assertAlmostEqual(fitted_min, self.uniform_data.min(), delta=0.1)
        self.assertAlmostEqual(fitted_max, self.uniform_data.max(), delta=0.1)

    def test_sample_distribution_properties(self):
        """Test that samples maintain distributional properties."""
        # Test with Gaussian
        gaussian = GaussianDistribution()
        gaussian.fit(self.normal_data)

        samples = gaussian.sample(1000)

        # Sample mean should be close to fitted mean
        self.assertAlmostEqual(np.mean(samples), gaussian.mean_[0], delta=1.0)
        # Sample std should be close to fitted std
        self.assertAlmostEqual(np.std(samples), np.sqrt(gaussian.cov_[0, 0]), delta=1.0)


if __name__ == "__main__":
    unittest.main()
