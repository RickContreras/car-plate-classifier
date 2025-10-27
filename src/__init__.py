"""
FC Detection - Redes Neuronales Fully Connected para Detección de Bounding Boxes

Un framework profesional para entrenar modelos de detección usando métodos
tradicionales de extracción de características (HOG, BRISK) con redes neuronales
completamente conectadas.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import features
from . import models
from . import data
from . import training
from . import evaluation

__all__ = ['features', 'models', 'data', 'training', 'evaluation']
