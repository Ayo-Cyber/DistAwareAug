"""
DistAwareAug: Distribution-Aware Data Augmentation for Imbalanced Learning

A Python package for intelligent oversampling that preserves the underlying
distribution of minority class features while ensuring sample diversity.
"""

from .augmentor import DistAwareAugmentor
from .config import DEFAULT_CONFIG
from .distance import DistanceMetrics
from .distribution import DistributionFitter
from .utils import check_class_balance, clip_to_range, validate_data

__version__ = "0.1.0"
__author__ = "Atunrase Ayo"
__email__ = "atunraseayomide@gmail.com"
__license__ = "MIT"

__all__ = [
    "DistAwareAugmentor",
    "DistributionFitter",
    "DistanceMetrics",
    "validate_data",
    "clip_to_range",
    "check_class_balance",
    "DEFAULT_CONFIG",
]
