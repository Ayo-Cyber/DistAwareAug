"""
Tests for the DistAwareAugmentor class.
"""

import unittest
import numpy as np
from sklearn.datasets import make_classification
from numpy.testing import assert_array_equal
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distawareaug import DistAwareAugmentor
from distawareaug.config import DEFAULT_CONFIG


class TestDistAwareAugmentor(unittest.TestCase):
    """Test cases for DistAwareAugmentor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple imbalanced dataset
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=4,
            n_redundant=0,
            n_informative=4,
            n_clusters_per_class=1,
            weights=[0.8, 0.2],
            random_state=42
        )
        
        # Initialize augmentor with default settings
        self.augmentor = DistAwareAugmentor(random_state=42)
        
    def test_initialization(self):
        """Test augmentor initialization with default parameters."""
        self.assertEqual(self.augmentor.sampling_strategy, 'auto')
        self.assertEqual(self.augmentor.diversity_threshold, 0.1)
        self.assertEqual(self.augmentor.distribution_method, 'kde')
        self.assertEqual(self.augmentor.distance_metric, 'euclidean')
        self.assertEqual(self.augmentor.random_state, 42)
        
    def test_initialization_with_custom_params(self):
        """Test augmentor initialization with custom parameters."""
        custom_augmentor = DistAwareAugmentor(
            sampling_strategy='minority',
            diversity_threshold=0.2,
            distribution_method='gaussian',
            distance_metric='manhattan',
            random_state=123
        )
        
        self.assertEqual(custom_augmentor.sampling_strategy, 'minority')
        self.assertEqual(custom_augmentor.diversity_threshold, 0.2)
        self.assertEqual(custom_augmentor.distribution_method, 'gaussian')
        self.assertEqual(custom_augmentor.distance_metric, 'manhattan')
        self.assertEqual(custom_augmentor.random_state, 123)
        
    def test_fit(self):
        """Test the fit method."""
        self.augmentor.fit(self.X, self.y)
        
        # Check that necessary attributes are set
        self.assertTrue(hasattr(self.augmentor, 'X_'))
        self.assertTrue(hasattr(self.augmentor, 'y_'))
        self.assertTrue(hasattr(self.augmentor, 'classes_'))
        self.assertTrue(hasattr(self.augmentor, 'class_counts_'))
        self.assertTrue(hasattr(self.augmentor, 'fitted_distributions_'))
        
        # Check shapes and values
        assert_array_equal(self.augmentor.X_, self.X)
        assert_array_equal(self.augmentor.y_, self.y)
        self.assertEqual(len(self.augmentor.classes_), 2)
        
    def test_fit_resample_auto_strategy(self):
        """Test fit_resample with auto sampling strategy."""
        X_resampled, y_resampled = self.augmentor.fit_resample(self.X, self.y)
        
        # Check that data was augmented
        self.assertGreater(len(X_resampled), len(self.X))
        self.assertGreater(len(y_resampled), len(self.y))
        
        # Check that classes are balanced
        unique_classes, class_counts = np.unique(y_resampled, return_counts=True)
        self.assertEqual(len(unique_classes), 2)
        self.assertEqual(class_counts[0], class_counts[1])  # Should be equal for 'auto'
        
        # Check that original data is preserved
        np.testing.assert_array_equal(X_resampled[:len(self.X)], self.X)
        np.testing.assert_array_equal(y_resampled[:len(self.y)], self.y)
        
    def test_fit_resample_shapes(self):
        """Test that fit_resample returns correct shapes."""
        X_resampled, y_resampled = self.augmentor.fit_resample(self.X, self.y)
        
        # Check shapes
        self.assertEqual(len(X_resampled), len(y_resampled))
        self.assertEqual(X_resampled.shape[1], self.X.shape[1])
        
    def test_different_distribution_methods(self):
        """Test different distribution methods."""
        methods = ['kde', 'gaussian', 'uniform']
        
        for method in methods:
            with self.subTest(method=method):
                augmentor = DistAwareAugmentor(
                    distribution_method=method,
                    random_state=42
                )
                
                X_resampled, y_resampled = augmentor.fit_resample(self.X, self.y)
                
                # Should generate some synthetic samples
                self.assertGreater(len(X_resampled), len(self.X))
                
    def test_different_distance_metrics(self):
        """Test different distance metrics."""
        metrics = ['euclidean', 'manhattan', 'cosine']
        
        for metric in metrics:
            with self.subTest(metric=metric):
                augmentor = DistAwareAugmentor(
                    distance_metric=metric,
                    random_state=42
                )
                
                try:
                    X_resampled, y_resampled = augmentor.fit_resample(self.X, self.y)
                    self.assertGreater(len(X_resampled), len(self.X))
                except Exception as e:
                    self.fail(f"Failed with {metric}: {e}")
                    
    def test_diversity_threshold_effect(self):
        """Test that diversity threshold affects sample generation."""
        thresholds = [0.01, 0.1, 0.5]
        sample_counts = []
        
        for threshold in thresholds:
            augmentor = DistAwareAugmentor(
                diversity_threshold=threshold,
                random_state=42
            )
            
            X_resampled, y_resampled = augmentor.fit_resample(self.X, self.y)
            sample_counts.append(len(X_resampled))
            
        # Higher threshold should generally result in fewer samples
        # (though this may not always be strictly monotonic due to randomness)
        self.assertGreaterEqual(sample_counts[0], sample_counts[-1])
        
    def test_random_state_reproducibility(self):
        """Test that random_state ensures reproducible results."""
        augmentor1 = DistAwareAugmentor(random_state=42)
        augmentor2 = DistAwareAugmentor(random_state=42)
        
        X_res1, y_res1 = augmentor1.fit_resample(self.X, self.y)
        X_res2, y_res2 = augmentor2.fit_resample(self.X, self.y)
        
        # Results should be identical
        np.testing.assert_array_equal(X_res1, X_res2)
        np.testing.assert_array_equal(y_res1, y_res2)
        
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        # Invalid distribution method
        with self.assertRaises(ValueError):
            augmentor = DistAwareAugmentor(distribution_method='invalid_method')
            augmentor.fit_resample(self.X, self.y)
            
        # Invalid distance metric
        with self.assertRaises(ValueError):
            augmentor = DistAwareAugmentor(distance_metric='invalid_metric')
            augmentor.fit_resample(self.X, self.y)
            
    def test_edge_cases(self):
        """Test edge cases."""
        # Single feature
        X_single = self.X[:, :1]
        augmentor = DistAwareAugmentor(random_state=42)
        X_res, y_res = augmentor.fit_resample(X_single, self.y)
        self.assertEqual(X_res.shape[1], 1)
        
        # Very small dataset
        X_small = self.X[:10]
        y_small = self.y[:10]
        augmentor = DistAwareAugmentor(random_state=42)
        X_res, y_res = augmentor.fit_resample(X_small, y_small)
        self.assertGreater(len(X_res), len(X_small))
        
    def test_custom_sampling_strategy(self):
        """Test custom sampling strategy."""
        # Custom strategy: generate 50 samples for class 1
        custom_strategy = {0: 0, 1: 50}
        augmentor = DistAwareAugmentor(
            sampling_strategy=custom_strategy,
            random_state=42
        )
        
        X_resampled, y_resampled = augmentor.fit_resample(self.X, self.y)
        
        # Check that the right number of samples were generated
        minority_count_orig = np.sum(self.y == 1)
        minority_count_new = np.sum(y_resampled == 1)
        
        # Should have approximately the requested increase
        # (may not be exact due to diversity constraints)
        self.assertGreater(minority_count_new, minority_count_orig)


class TestAugmentorIntegration(unittest.TestCase):
    """Integration tests for DistAwareAugmentor."""
    
    def test_sklearn_compatibility(self):
        """Test compatibility with sklearn pipeline and cross-validation."""
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        
        # Create dataset
        X, y = make_classification(
            n_samples=200,
            n_features=4,
            n_redundant=0,
            weights=[0.7, 0.3],
            random_state=42
        )
        
        # Create pipeline (Note: DistAwareAugmentor doesn't transform, so we test fit_resample)
        augmentor = DistAwareAugmentor(random_state=42)
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Test that it works with the workflow
        X_aug, y_aug = augmentor.fit_resample(X, y)
        classifier.fit(X_aug, y_aug)
        
        # Should be able to predict
        predictions = classifier.predict(X)
        self.assertEqual(len(predictions), len(X))
        

if __name__ == '__main__':
    unittest.main()