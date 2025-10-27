"""
Módulo de extracción de características para FC Detection Project.
"""

from .base import FeatureExtractor
from .hog import HOGFeatureExtractor
from .brisk import BRISKFeatureExtractor

__all__ = ['FeatureExtractor', 'HOGFeatureExtractor', 'BRISKFeatureExtractor']
