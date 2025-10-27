"""
Módulo de evaluación para modelos de detección.
"""

from .metrics import calculate_iou, evaluate_detections, MetricsCalculator

__all__ = ['calculate_iou', 'evaluate_detections', 'MetricsCalculator']
