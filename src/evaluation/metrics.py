"""
Métricas para evaluación de detección.
"""

import numpy as np
from typing import Tuple, List, Dict


def calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """
    Calcular Intersection over Union (IoU) para dos bounding boxes normalizados.
    
    Args:
        box1: (x_center, y_center, width, height) normalizado
        box2: (x_center, y_center, width, height) normalizado
        
    Returns:
        Puntuación IoU entre 0 y 1
    """
    # Convertir de formato centro a formato de esquinas
    box1_x1 = box1[0] - box1[2] / 2
    box1_y1 = box1[1] - box1[3] / 2
    box1_x2 = box1[0] + box1[2] / 2
    box1_y2 = box1[1] + box1[3] / 2
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2
    
    # Calcular intersección
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calcular unión
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def evaluate_detections(y_true: np.ndarray, y_pred: np.ndarray, iou_threshold: float = 0.5) -> Dict:
    """
    Evaluar predicciones de detección.
    
    Args:
        y_true: Cajas verdaderas (n_samples, 4)
        y_pred: Cajas predichas (n_samples, 4)
        iou_threshold: Umbral de IoU para considerar una detección correcta
        
    Returns:
        Diccionario con métricas de evaluación
    """
    assert len(y_true) == len(y_pred), "y_true y y_pred deben tener la misma longitud"
    
    iou_scores = []
    mae_scores = []
    correct_detections = 0
    
    for true_box, pred_box in zip(y_true, y_pred):
        # Calcular IoU
        iou = calculate_iou(pred_box, true_box)
        iou_scores.append(iou)
        
        # Calcular MAE (Error Absoluto Medio)
        mae = np.mean(np.abs(pred_box - true_box))
        mae_scores.append(mae)
        
        # Contar detecciones correctas
        if iou >= iou_threshold:
            correct_detections += 1
    
    metrics = {
        'avg_iou': np.mean(iou_scores),
        'median_iou': np.median(iou_scores),
        'std_iou': np.std(iou_scores),
        'min_iou': np.min(iou_scores),
        'max_iou': np.max(iou_scores),
        'mae': np.mean(mae_scores),
        'accuracy': correct_detections / len(y_true),  # Porcentaje con IoU > umbral
        'total_samples': len(y_true),
        'correct_detections': correct_detections
    }
    
    return metrics


class MetricsCalculator:
    """Clase para calcular y rastrear métricas de detección."""
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75, 0.9]):
        """
        Inicializar calculador de métricas.
        
        Args:
            iou_thresholds: Lista de umbrales de IoU a evaluar
        """
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        """Reiniciar todas las métricas."""
        self.iou_scores = []
        self.mae_scores = []
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Actualizar métricas con nuevas predicciones.
        
        Args:
            y_true: Cajas verdaderas
            y_pred: Cajas predichas
        """
        for true_box, pred_box in zip(y_true, y_pred):
            iou = calculate_iou(pred_box, true_box)
            mae = np.mean(np.abs(pred_box - true_box))
            
            self.iou_scores.append(iou)
            self.mae_scores.append(mae)
    
    def compute(self) -> Dict:
        """
        Computar métricas finales.
        
        Returns:
            Diccionario de métricas
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
        
        # Agregar precisión en diferentes umbrales de IoU
        for threshold in self.iou_thresholds:
            correct = np.sum(iou_array >= threshold)
            accuracy = correct / len(iou_array)
            metrics[f'accuracy@{threshold}'] = float(accuracy)
            metrics[f'correct@{threshold}'] = int(correct)
        
        return metrics
    
    def print_metrics(self, metrics: Dict = None):
        """Imprimir métricas en formato formateado."""
        if metrics is None:
            metrics = self.compute()
        
        print("\n" + "="*60)
        print("Métricas de Detección")
        print("="*60)
        
        print(f"Total de muestras: {metrics['total_samples']}")
        print(f"\nEstadísticas de IoU:")
        print(f"  IoU Promedio: {metrics['avg_iou']:.4f}")
        print(f"  IoU Mediana:  {metrics['median_iou']:.4f}")
        print(f"  Desv. Est. IoU:  {metrics['std_iou']:.4f}")
        print(f"  IoU Mínimo:   {metrics['min_iou']:.4f}")
        print(f"  IoU Máximo:   {metrics['max_iou']:.4f}")
        
        print(f"\nMAE: {metrics['mae']:.4f}")
        
        print(f"\nPrecisión en diferentes umbrales de IoU:")
        for threshold in self.iou_thresholds:
            acc_key = f'accuracy@{threshold}'
            correct_key = f'correct@{threshold}'
            if acc_key in metrics:
                print(f"  IoU >= {threshold}: {metrics[acc_key]:.2%} ({metrics[correct_key]}/{metrics['total_samples']})")
        
        print("="*60 + "\n")
