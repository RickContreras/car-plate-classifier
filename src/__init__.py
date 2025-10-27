"""
FC Detection - Fully Connected Neural Networks for Bounding Box Detection

A professional framework for training detection models using traditional
feature extraction methods (HOG, BRISK) with fully connected neural networks.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import features
from . import models
from . import data
from . import training
from . import evaluation

__all__ = ['features', 'models', 'data', 'training', 'evaluation']
