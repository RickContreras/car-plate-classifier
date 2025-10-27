"""
Módulo de datos para carga y preprocesamiento de datasets.
"""

from .dataset import DetectionDataset, load_dataset
from .utils import normalize_bbox, denormalize_bbox

__all__ = ['DetectionDataset', 'load_dataset', 'normalize_bbox', 'denormalize_bbox']
