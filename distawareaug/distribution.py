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
    
    def __init__(self, bandwidth: str = 'scott', kernel: str = 'gaussian', random_state: Optional[int] = None):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.kde_ = None
        self.random_state = random_state
        
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
        samples = self.kde_.sample(n_samples, random_state=self.random_state)
        if self.n_features_ == 1:
            samples = samples.flatten()
        return samples


class GaussianDistribution(BaseDistribution):
    """Multivariate Gaussian distribution."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.mean_ = None
        self.cov_ = None
        self.random_state = random_state
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random
        
    def fit(self, data: np.ndarray) -> 'GaussianDistribution':
        """Fit Gaussian distribution to the data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.mean_ = np.mean(data, axis=0)
        # Use ddof=0 for population covariance (matches np.var behavior)
        self.cov_ = np.cov(data, rowvar=False, ddof=0)
        
        # Handle 1D case where cov_ is a scalar
        if self.cov_.ndim == 0:
            self.cov_ = self.cov_.reshape(1, 1)
        
        # Handle singular covariance matrices
        if self.cov_.shape[0] > 0 and np.linalg.det(self.cov_) == 0:
            self.cov_ += np.eye(self.cov_.shape[0]) * 1e-6
            
        self.n_features_ = data.shape[1]
        return self
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the fitted Gaussian."""
        if self.n_features_ == 1:
            samples = self.rng.normal(
                self.mean_[0], 
                np.sqrt(self.cov_[0, 0]), 
                n_samples
            )
        else:
            samples = self.rng.multivariate_normal(
                self.mean_, 
                self.cov_, 
                n_samples
            )
        return samples


class UniformDistribution(BaseDistribution):
    """Uniform distribution within feature bounds."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.bounds_ = None
        self.random_state = random_state
        if random_state is not None:
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random
        
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
        samples = self.rng.uniform(
            self.bounds_[:, 0],
            self.bounds_[:, 1],
            size=(n_samples, self.n_features_)
        )
        
        if self.n_features_ == 1:
            samples = samples.flatten()
            
        return samples


class CompositeDistribution(BaseDistribution):
    """
    A composite distribution that samples from multiple independent distributions.
    """
    def __init__(self, distributions: Dict[int, BaseDistribution]):
        self.distributions = distributions

    def __len__(self) -> int:
        """Return the number of features (distributions)."""
        return len(self.distributions)
    
    def __getitem__(self, key: int) -> BaseDistribution:
        """Allow subscript access to individual distributions."""
        return self.distributions[key]
    
    def __contains__(self, key: int) -> bool:
        """Check if a feature index has a distribution."""
        return key in self.distributions
    
    def __iter__(self):
        """Iterate over feature indices."""
        return iter(self.distributions)

    def fit(self, data: np.ndarray) -> 'CompositeDistribution':
        """This distribution is already fitted."""
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the composite distribution."""
        n_features = len(self.distributions)
        samples = np.zeros((n_samples, n_features))

        for i, distribution in self.distributions.items():
            feature_samples = distribution.sample(n_samples)
            samples[:, i] = feature_samples

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
            
    def fit(self, data: np.ndarray) -> CompositeDistribution:
        """
        Fit distributions to each feature independently.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Data to fit distributions to
            
        Returns
        -------
        distributions : CompositeDistribution
            A composite distribution containing fitted distributions for each feature.
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_features = data.shape[1]
        distributions = {}
        
        for i in range(n_features):
            feature_data = data[:, i]
            
            # Create distribution instance with random_state
            if self.method == 'kde':
                dist = KDEDistribution(random_state=self.random_state, **self.kwargs)
            elif self.method == 'gaussian':
                dist = GaussianDistribution(random_state=self.random_state, **self.kwargs)
            elif self.method == 'uniform':
                dist = UniformDistribution(random_state=self.random_state, **self.kwargs)
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            # Fit to feature data
            distributions[i] = dist.fit(feature_data)
        
        # Store fitted distributions for sample() method
        self.fitted_distributions_ = CompositeDistribution(distributions)
        return self.fitted_distributions_
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Generate samples from the fitted distributions.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        Returns
        -------
        samples : array-like of shape (n_samples, n_features)
            Generated samples
            
        Raises
        ------
        ValueError
            If called before fit()
        """
        if not hasattr(self, 'fitted_distributions_'):
            raise ValueError("Must call fit() before sample()")
        
        return self.fitted_distributions_.sample(n_samples)
        
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
        composite_dist = self.fit(data)
        self.fitted_distributions_ = composite_dist
        return composite_dist.sample(n_samples)