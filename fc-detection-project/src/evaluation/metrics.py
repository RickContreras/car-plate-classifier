"""
Metrics for detection evaluation.
"""

import numpy as np
from typing import Tuple, List, Dict


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) for two normalized bounding boxes.
    
    Args:
        box1: (x_center, y_center, width, height) normalized
        box2: (x_center, y_center, width, height) normalized
        
    Returns:
        IoU score between 0 and 1
    """
    # Convert from center format to corner format
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_detections(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate detection predictions.
    
    Args:
        y_true: Ground truth boxes (n_samples, 4)
        y_pred: Predicted boxes (n_samples, 4)
        iou_threshold: IoU threshold for considering a detection correct
        
    Returns:
        Dictionary with evaluation metrics
    """
    assert len(y_true) == len(y_pred), "y_true and y_pred must have same length"
    
    iou_scores = []
    mae_scores = []
    correct_detections = 0
    
    for true_box, pred_box in zip(y_true, y_pred):
        # Calculate IoU
        iou = calculate_iou(pred_box, true_box)
        iou_scores.append(iou)
        
        # Calculate MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred_box - true_box))
        mae_scores.append(mae)
        
        # Count correct detections
        if iou >= iou_threshold:
            correct_detections += 1
    
    metrics = {
        'avg_iou': np.mean(iou_scores),
        'median_iou': np.median(iou_scores),
        'std_iou': np.std(iou_scores),
        'min_iou': np.min(iou_scores),
        'max_iou': np.max(iou_scores),
        'mae': np.mean(mae_scores),
        'accuracy': correct_detections / len(y_true),  # Percentage with IoU > threshold
        'total_samples': len(y_true),
        'correct_detections': correct_detections
    }
    
    return metrics


class MetricsCalculator:
    """Class to calculate and track detection metrics."""
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75, 0.9]):
        """
        Initialize metrics calculator.
        
        Args:
            iou_thresholds: List of IoU thresholds to evaluate
        """
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.iou_scores = []
        self.mae_scores = []
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update metrics with new predictions.
        
        Args:
            y_true: Ground truth boxes
            y_pred: Predicted boxes
        """
        for true_box, pred_box in zip(y_true, y_pred):
            iou = calculate_iou(pred_box, true_box)
            mae = np.mean(np.abs(pred_box - true_box))
            
            self.iou_scores.append(iou)
            self.mae_scores.append(mae)
    
    def compute(self) -> Dict:
        """
        Compute final metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.iou_scores) == 0:
            return {}
        
        iou_array = np.array(self.iou_scores)
        
        metrics = {
            'avg_iou': float(np.mean(iou_array)),
            'median_iou': float(np.median(iou_array)),
            'std_iou': float(np.std(iou_array)),
            'min_iou': float(np.min(iou_array)),
            'max_iou': float(np.max(iou_array)),
            'mae': float(np.mean(self.mae_scores)),
            'total_samples': len(self.iou_scores)
        }
        
        # Add accuracy at different IoU thresholds
        for threshold in self.iou_thresholds:
            correct = np.sum(iou_array >= threshold)
            accuracy = correct / len(iou_array)
            metrics[f'accuracy@{threshold}'] = float(accuracy)
            metrics[f'correct@{threshold}'] = int(correct)
        
        return metrics
    
    def print_metrics(self, metrics: Dict = None):
        """Print metrics in a formatted way."""
        if metrics is None:
            metrics = self.compute()
        
        print("\n" + "="*60)
        print("Detection Metrics")
        print("="*60)
        
        print(f"Total samples: {metrics['total_samples']}")
        print(f"\nIoU Statistics:")
        print(f"  Average IoU: {metrics['avg_iou']:.4f}")
        print(f"  Median IoU:  {metrics['median_iou']:.4f}")
        print(f"  Std IoU:     {metrics['std_iou']:.4f}")
        print(f"  Min IoU:     {metrics['min_iou']:.4f}")
        print(f"  Max IoU:     {metrics['max_iou']:.4f}")
        
        print(f"\nMAE: {metrics['mae']:.4f}")
        
        print(f"\nAccuracy at different IoU thresholds:")
        for threshold in self.iou_thresholds:
            acc_key = f'accuracy@{threshold}'
            correct_key = f'correct@{threshold}'
            if acc_key in metrics:
                print(f"  IoU >= {threshold}: {metrics[acc_key]:.2%} ({metrics[correct_key]}/{metrics['total_samples']})")
        
        print("="*60 + "\n")
