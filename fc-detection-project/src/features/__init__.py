"""
Feature extraction module for FC Detection Project.
"""

from .base import FeatureExtractor
from .hog import HOGFeatureExtractor
from .brisk import BRISKFeatureExtractor

__all__ = ['FeatureExtractor', 'HOGFeatureExtractor', 'BRISKFeatureExtractor']
