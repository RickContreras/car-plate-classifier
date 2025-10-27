"""
Módulo de entrenamiento para modelos de detección.
"""

from .trainer import Trainer
from .callbacks import get_callbacks

__all__ = ['Trainer', 'get_callbacks']
