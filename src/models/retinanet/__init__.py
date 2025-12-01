"""
Módulo RetinaNet para detección de objetos.

Este módulo implementa la arquitectura RetinaNet con:
- Feature Pyramid Network (FPN)
- Focal Loss para class imbalance
- Anchor-based detection
"""

from .detector import RetinaNetDetector
from .anchors import AnchorGenerator
from .losses import FocalLoss, SmoothL1Loss, RetinaNetLoss
from .backbone import ResNetBackbone
from .fpn import FeaturePyramidNetwork

__all__ = [
    'RetinaNetDetector',
    'AnchorGenerator',
    'FocalLoss',
    'SmoothL1Loss',
    'RetinaNetLoss',
    'ResNetBackbone',
    'FeaturePyramidNetwork'
]
