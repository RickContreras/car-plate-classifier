"""
Evaluation module for detection models.
"""

from .metrics import calculate_iou, evaluate_detections, MetricsCalculator

__all__ = ['calculate_iou', 'evaluate_detections', 'MetricsCalculator']
