"""
Default configuration parameters and constants for DistAwareAug.
"""

# Default configuration for DistAwareAugmentor
DEFAULT_CONFIG = {
    # Sampling strategy
    'sampling_strategy': 'auto',
    
    # Diversity parameters
    'diversity_threshold': 0.1,
    'adaptive_threshold': True,
    'max_generation_attempts': 1000,
    
    # Distribution fitting
    'distribution_method': 'kde',
    'kde_bandwidth': 'scott',
    'kde_kernel': 'gaussian',
    
    # Distance metrics
    'distance_metric': 'euclidean',
    'distance_kwargs': {},
    
    # Feature processing
    'normalize_features': False,
    'normalization_method': 'standardize',
    'clip_method': 'minmax',
    'clip_margin': 0.0,
    
    # Categorical feature handling
    'handle_categorical': True,
    'categorical_threshold': 0.1,
    'max_categorical_values': 20,
    
    # Random state
    'random_state': None,
    
    # Validation
    'validate_input': True,
    'min_samples_per_class': 2,
    
    # Performance
    'n_jobs': 1,
    'verbose': False
}

# Distribution method parameters
DISTRIBUTION_PARAMS = {
    'kde': {
        'bandwidth': ['scott', 'silverman', 0.1, 0.5, 1.0],
        'kernel': ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    },
    'gaussian': {
        'regularization': [1e-6, 1e-5, 1e-4, 1e-3]
    },
    'uniform': {
        'margin': [0.0, 0.05, 0.1, 0.2]
    }
}

# Distance metric parameters
DISTANCE_PARAMS = {
    'euclidean': {},
    'manhattan': {},
    'cosine': {},
    'minkowski': {
        'p': [1, 2, 3, 4, 5]
    },
    'chebyshev': {},
    'hamming': {},
    'jaccard': {}
}

# Feature normalization methods
NORMALIZATION_METHODS = [
    'standardize',  # Z-score normalization
    'minmax',       # Min-max scaling
    'robust'        # Robust scaling using median and MAD
]

# Clipping methods for synthetic samples
CLIPPING_METHODS = [
    'minmax',       # Clip to min/max of reference data
    'percentile',   # Clip to 5th-95th percentile
    'std'           # Clip to mean ± 3 standard deviations
]

# Supported sampling strategies
SAMPLING_STRATEGIES = [
    'auto',         # Balance all classes to majority
    'minority',     # Oversample only minority class
    'not_minority', # Oversample all except majority
    'all'           # Oversample all classes
]

# Quality metrics for evaluating synthetic samples
QUALITY_METRICS = {
    'diversity': {
        'internal_diversity': 'Average pairwise distance within synthetic samples',
        'external_diversity': 'Average distance to original samples',
        'nearest_neighbor_distance': 'Distance to nearest original sample'
    },
    'distribution': {
        'ks_test': 'Kolmogorov-Smirnov test for distribution similarity',
        'wasserstein_distance': 'Earth mover\'s distance between distributions',
        'jensen_shannon_divergence': 'Jensen-Shannon divergence'
    },
    'classification': {
        'accuracy_improvement': 'Improvement in classifier accuracy',
        'f1_score_improvement': 'Improvement in F1 score',
        'auc_improvement': 'Improvement in AUC-ROC'
    }
}

# Warning thresholds
WARNING_THRESHOLDS = {
    'min_samples_per_class': 2,
    'max_imbalance_ratio': 100,
    'min_diversity_threshold': 1e-6,
    'max_generation_attempts': 10000,
    'max_features': 1000
}

# Default hyperparameters for different dataset sizes
SIZE_BASED_CONFIG = {
    'small': {  # < 1000 samples
        'distribution_method': 'gaussian',
        'diversity_threshold': 0.05,
        'max_generation_attempts': 500
    },
    'medium': {  # 1000 - 10000 samples
        'distribution_method': 'kde',
        'diversity_threshold': 0.1,
        'max_generation_attempts': 1000
    },
    'large': {  # > 10000 samples
        'distribution_method': 'kde',
        'diversity_threshold': 0.15,
        'max_generation_attempts': 2000
    }
}

# Dataset complexity based configurations
COMPLEXITY_BASED_CONFIG = {
    'low': {  # Few features, simple distributions
        'distribution_method': 'gaussian',
        'distance_metric': 'euclidean',
        'normalize_features': False
    },
    'medium': {  # Moderate features, mixed distributions
        'distribution_method': 'kde',
        'distance_metric': 'euclidean',
        'normalize_features': True
    },
    'high': {  # Many features, complex distributions
        'distribution_method': 'kde',
        'distance_metric': 'cosine',
        'normalize_features': True,
        'adaptive_threshold': True
    }
}

# Experimental configurations for research
EXPERIMENTAL_CONFIG = {
    'conservative': {
        'diversity_threshold': 0.2,
        'distribution_method': 'gaussian',
        'clip_margin': 0.0
    },
    'aggressive': {
        'diversity_threshold': 0.05,
        'distribution_method': 'kde',
        'clip_margin': 0.1
    },
    'balanced': {
        'diversity_threshold': 0.1,
        'distribution_method': 'kde',
        'clip_margin': 0.05
    }
}

def get_config_for_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    imbalance_ratio: float = None
) -> dict:
    """
    Get recommended configuration based on dataset characteristics.
    
    Parameters
    ----------
    n_samples : int
        Number of samples in dataset
    n_features : int
        Number of features
    n_classes : int
        Number of classes
    imbalance_ratio : float, optional
        Ratio of majority to minority class
        
    Returns
    -------
    config : dict
        Recommended configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    # Size-based adjustments
    if n_samples < 1000:
        config.update(SIZE_BASED_CONFIG['small'])
    elif n_samples < 10000:
        config.update(SIZE_BASED_CONFIG['medium'])
    else:
        config.update(SIZE_BASED_CONFIG['large'])
        
    # Complexity-based adjustments
    if n_features < 10:
        config.update(COMPLEXITY_BASED_CONFIG['low'])
    elif n_features < 50:
        config.update(COMPLEXITY_BASED_CONFIG['medium'])
    else:
        config.update(COMPLEXITY_BASED_CONFIG['high'])
        
    # Imbalance-based adjustments
    if imbalance_ratio and imbalance_ratio > 10:
        config['diversity_threshold'] = max(0.05, config['diversity_threshold'] * 0.5)
        config['max_generation_attempts'] *= 2
        
    # High-dimensional datasets
    if n_features > 100:
        config['normalize_features'] = True
        config['distance_metric'] = 'cosine'
        
    return config