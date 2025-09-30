"""
Distribution fitting and sampling for feature distributions.
"""

import numpy as np
from typing import Optional, Union, Dict, Any
from sklearn.neighbors import KernelDensity
from scipy import stats
from abc import ABC, abstractmethod


class BaseDistribution(ABC):
    """Base class for distribution fitting and sampling."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BaseDistribution':
        """Fit the distribution to data."""
        pass
        
    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the fitted distribution."""
        pass


class KDEDistribution(BaseDistribution):
    """Kernel Density Estimation distribution."""
    
    def __init__(self, bandwidth: str = 'scott', kernel: str = 'gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_ = None
        
    def fit(self, data: np.ndarray) -> 'KDEDistribution':
        """Fit KDE to the data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel
        ).fit(data)
        
        self.n_features_ = data.shape[1]
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the fitted KDE."""
        samples = self.kde_.sample(n_samples)
        if self.n_features_ == 1:
            samples = samples.flatten()
        return samples


class GaussianDistribution(BaseDistribution):
    """Multivariate Gaussian distribution."""
    
    def __init__(self):
        self.mean_ = None
        self.cov_ = None
        
    def fit(self, data: np.ndarray) -> 'GaussianDistribution':
        """Fit Gaussian distribution to the data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.mean_ = np.mean(data, axis=0)
        self.cov_ = np.cov(data, rowvar=False)
        
        # Handle singular covariance matrices
        if np.linalg.det(self.cov_) == 0:
            self.cov_ += np.eye(self.cov_.shape[0]) * 1e-6
            
        self.n_features_ = data.shape[1]
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the fitted Gaussian."""
        if self.n_features_ == 1:
            samples = np.random.normal(
                self.mean_[0], 
                np.sqrt(self.cov_[0, 0]), 
                n_samples
            )
        else:
            samples = np.random.multivariate_normal(
                self.mean_, 
                self.cov_, 
                n_samples
            )
        return samples


class UniformDistribution(BaseDistribution):
    """Uniform distribution within feature bounds."""
    
    def __init__(self):
        self.bounds_ = None
        
    def fit(self, data: np.ndarray) -> 'UniformDistribution':
        """Fit uniform distribution to data bounds."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.bounds_ = np.column_stack([
            np.min(data, axis=0),
            np.max(data, axis=0)
        ])
        
        self.n_features_ = data.shape[1]
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate uniform samples within bounds."""
        samples = np.random.uniform(
            self.bounds_[:, 0],
            self.bounds_[:, 1],
            size=(n_samples, self.n_features_)
        )
        
        if self.n_features_ == 1:
            samples = samples.flatten()
            
        return samples


class DistributionFitter:
    """
    Main class for fitting and sampling from feature distributions.
    
    Parameters
    ----------
    method : str, default='kde'
        Distribution fitting method ('kde', 'gaussian', 'uniform')
    random_state : int, default=None
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for specific distribution methods
    """
    
    def __init__(
        self, 
        method: str = 'kde',
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.method = method
        self.random_state = random_state
        self.kwargs = kwargs
        
        if random_state is not None:
            np.random.seed(random_state)
            
        self.distributions_ = {}
        
    def fit(self, data: np.ndarray) -> Dict[int, BaseDistribution]:
        """
        Fit distributions to each feature independently.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Data to fit distributions to
            
        Returns
        -------
        distributions : dict
            Dictionary mapping feature index to fitted distribution
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_features = data.shape[1]
        distributions = {}
        
        for i in range(n_features):
            feature_data = data[:, i]
            
            # Create distribution instance
            if self.method == 'kde':
                dist = KDEDistribution(**self.kwargs)
            elif self.method == 'gaussian':
                dist = GaussianDistribution(**self.kwargs)
            elif self.method == 'uniform':
                dist = UniformDistribution(**self.kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            # Fit to feature data
            distributions[i] = dist.fit(feature_data)
            
        self.distributions_ = distributions
        return distributions
        
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from fitted distributions.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        samples : array-like of shape (n_samples, n_features)
            Generated samples
        """
        if not self.distributions_:
            raise ValueError("Must fit distributions before sampling")
            
        n_features = len(self.distributions_)
        samples = np.zeros((n_samples, n_features))
        
        for i, distribution in self.distributions_.items():
            feature_samples = distribution.sample(n_samples)
            samples[:, i] = feature_samples
            
        return samples
        
    def fit_sample(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Fit distributions and generate samples in one step.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Data to fit distributions to
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        samples : array-like of shape (n_samples, n_features)
            Generated samples
        """
        self.fit(data)
        return self.sample(n_samples)