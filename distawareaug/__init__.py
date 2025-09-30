"""
DistAwareAug: Distribution-Aware Data Augmentation for Imbalanced Learning

A Python package for intelligent oversampling that preserves the underlying 
distribution of minority class features while ensuring sample diversity.
"""

from .augmentor import DistAwareAugmentor
from .distribution import DistributionFitter
from .distance import DistanceMetrics
from .utils import validate_data, clip_to_range
from .config import DEFAULT_CONFIG

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "DistAwareAugmentor",
    "DistributionFitter", 
    "DistanceMetrics",
    "validate_data",
    "clip_to_range",
    "DEFAULT_CONFIG"
]