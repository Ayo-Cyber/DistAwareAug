"""
Main oversampling logic for distribution-aware augmentation.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y

from .distribution import DistributionFitter
from .distance import DistanceMetrics
from .utils import validate_data, clip_to_range
from .config import DEFAULT_CONFIG


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
        sampling_strategy: str = 'auto',
        diversity_threshold: float = 0.1,
        distribution_method: str = 'kde',
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.sampling_strategy = sampling_strategy
        self.diversity_threshold = diversity_threshold
        self.distribution_method = distribution_method
        self.distance_metric = distance_metric
        self.random_state = random_state
        
        # Initialize components
        self.distribution_fitter = DistributionFitter(
            method=distribution_method,
            random_state=random_state
        )
        self.distance_metrics = DistanceMetrics(metric=distance_metric)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DistAwareAugmentor':
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
        
        # Determine which classes need augmentation
        self._compute_sampling_strategy()
        
        # Fit distributions for minority classes
        self.fitted_distributions_ = {}
        for class_label in self.classes_to_augment_:
            class_mask = y == class_label
            class_data = X[class_mask]
            
            # Create a new DistributionFitter instance for each class
            class_fitter = DistributionFitter(
                method=self.distribution_method,
                random_state=self.random_state
            )
            class_fitter.fit(class_data)
            self.fitted_distributions_[class_label] = class_fitter
            
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
        Generate synthetic samples for minority classes.
        
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
        
        return X_resampled, y_resampled
        
    def _compute_sampling_strategy(self):
        """Compute how many samples to generate for each class."""
        if self.sampling_strategy == 'auto':
            # Balance all classes to majority class count
            max_count = np.max(self.class_counts_)
            self.sampling_strategy_ = {}
            
            for class_label, count in zip(self.classes_, self.class_counts_):
                self.sampling_strategy_[class_label] = max_count - count
                
        else:
            # Handle custom sampling strategies
            self.sampling_strategy_ = self.sampling_strategy
            
        # Identify classes that need augmentation
        self.classes_to_augment_ = [
            cls for cls, n_samples in self.sampling_strategy_.items()
            if n_samples > 0
        ]
        
    def _generate_samples(self, class_label: int, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a specific class."""
        class_mask = self.y_ == class_label
        class_data = self.X_[class_mask]
        
        # Get the fitted distribution fitter for this class
        distribution_fitter = self.fitted_distributions_[class_label]
        
        synthetic_samples = []
        attempts = 0
        max_attempts = n_samples * 10  # Avoid infinite loops
        
        while len(synthetic_samples) < n_samples and attempts < max_attempts:
            # Sample from the fitted distribution
            candidate = distribution_fitter.sample(1)[0]
            
            # Clip to feature ranges
            candidate = clip_to_range(candidate, class_data)
            
            # Check diversity constraint
            if self._is_diverse_enough(candidate, class_data, synthetic_samples):
                synthetic_samples.append(candidate)
                
            attempts += 1
            
        # If we couldn't generate enough diverse samples, fill with random samples
        while len(synthetic_samples) < n_samples:
            candidate = distribution_fitter.sample(1)[0]
            candidate = clip_to_range(candidate, class_data)
            synthetic_samples.append(candidate)
            
        return np.array(synthetic_samples)
        
    def _is_diverse_enough(
        self, 
        candidate: np.ndarray, 
        class_data: np.ndarray,
        synthetic_samples: list
    ) -> bool:
        """Check if a candidate sample is diverse enough."""
        # Check distance to original samples
        distances_to_original = self.distance_metrics.compute_distances(
            candidate.reshape(1, -1), class_data
        )[0]
        
        if np.min(distances_to_original) < self.diversity_threshold:
            return False
            
        # Check distance to already generated synthetic samples
        if synthetic_samples:
            synthetic_array = np.array(synthetic_samples)
            distances_to_synthetic = self.distance_metrics.compute_distances(
                candidate.reshape(1, -1), synthetic_array
            )[0]
            
            if np.min(distances_to_synthetic) < self.diversity_threshold:
                return False
                
        return True