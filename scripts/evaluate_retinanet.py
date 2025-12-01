#!/usr/bin/env python3
"""
Script para evaluar modelo RetinaNet en conjunto de test.

Calcula métricas de detección: mAP, precision, recall, IoU.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
import json
from typing import List, Tuple

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.retinanet_dataset import RetinaNetDataset
from src.models.retinanet.anchors import decode_boxes
from src.evaluation.metrics import calculate_iou


def load_config(config_path: str) -> dict:
    """Cargar configuración desde YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_path: str) -> tf.keras.Model:
    """
    Cargar modelo entrenado.
    
    Args:
        model_path: Ruta al modelo .h5
        
    Returns:
        Modelo cargado
    """
    print(f"📦 Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Modelo cargado exitosamente")
    return model


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
    max_detections: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplicar Non-Maximum Suppression.
    
    Args:
        boxes: Array de boxes (N, 4) en formato (x1, y1, x2, y2)
        scores: Array de scores (N,)
        iou_threshold: Umbral de IoU para suprimir
        score_threshold: Umbral mínimo de score
        max_detections: Máximo número de detecciones
        
    Returns:
        Tupla de (boxes_filtradas, scores_filtrados)
    """
    # Filtrar por score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    
    if len(boxes) == 0:
        return np.array([]), np.array([])
    
    # Ordenar por score descendente
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    
    # NMS
    keep = []
    while len(boxes) > 0 and len(keep) < max_detections:
        # Tomar la box con mayor score
        keep.append(0)
        
        if len(boxes) == 1:
            break
        
        # Calcular IoU con el resto
        current_box = boxes[0:1]
        other_boxes = boxes[1:]
        
        ious = []
        for other_box in other_boxes:
            # Convertir formato para calculate_iou
            # De (x1, y1, x2, y2) a (x_center, y_center, w, h) normalizado
            x1, y1, x2, y2 = current_box[0]
            w, h = x2 - x1, y2 - y1
            box1 = ((x1 + x2) / 2 / 640, (y1 + y2) / 2 / 640, w / 640, h / 640)
            
            x1, y1, x2, y2 = other_box
            w, h = x2 - x1, y2 - y1
            box2 = ((x1 + x2) / 2 / 640, (y1 + y2) / 2 / 640, w / 640, h / 640)
            
            iou = calculate_iou(box1, box2)
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Mantener boxes con IoU < threshold
        mask = ious < iou_threshold
        boxes = other_boxes[mask]
        scores = scores[1:][mask]
    
    keep_indices = np.array(keep)
    
    return boxes[keep_indices] if len(keep_indices) > 0 else np.array([]), \
           scores[keep_indices] if len(keep_indices) > 0 else np.array([])


def evaluate_model(
    model: tf.keras.Model,
    dataset: RetinaNetDataset,
    config: dict
) -> dict:
    """
    Evaluar modelo en dataset.
    
    Args:
        model: Modelo a evaluar
        dataset: Dataset de evaluación
        config: Configuración
        
    Returns:
        Diccionario con métricas
    """
    print("\n" + "="*70)
    print("EVALUANDO MODELO")
    print("="*70)
    
    all_ious = []
    num_correct_detections = {0.5: 0, 0.75: 0, 0.9: 0}
    total_detections = 0
    
    # Evaluar cada imagen
    for idx in range(len(dataset)):
        # Obtener imagen y ground truth
        image, (_, _, _) = dataset[idx]
        gt_boxes = dataset.annotations[idx]
        
        # Expandir batch dimension
        image_batch = np.expand_dims(image, axis=0)
        
        # Predecir
        cls_pred, box_pred = model.predict(image_batch, verbose=0)
        
        # Aplicar sigmoid a clasificación
        cls_pred = tf.nn.sigmoid(cls_pred[0]).numpy()
        box_pred = box_pred[0]
        
        # Decodificar boxes
        decoded_boxes = decode_boxes(
            tf.constant(box_pred, dtype=tf.float32),
            dataset.anchors
        ).numpy()
        
        # Obtener scores de clase "placa"
        scores = cls_pred[:, 0]
        
        # Aplicar NMS
        nms_config = config['evaluation']['nms']
        final_boxes, final_scores = non_max_suppression(
            decoded_boxes,
            scores,
            iou_threshold=nms_config['iou_threshold'],
            score_threshold=nms_config['score_threshold'],
            max_detections=nms_config['max_detections']
        )
        
        if len(final_boxes) == 0:
            continue
        
        # Tomar la detección con mayor score
        best_box = final_boxes[0]
        
        # Calcular IoU con ground truth
        if len(gt_boxes) > 0:
            gt_box = gt_boxes[0]  # Asumimos una sola placa por imagen
            
            # Convertir a formato normalizado
            x1, y1, x2, y2 = best_box
            w, h = x2 - x1, y2 - y1
            pred_box_norm = ((x1 + x2) / 2 / 640, (y1 + y2) / 2 / 640, w / 640, h / 640)
            
            x1, y1, x2, y2 = gt_box
            w, h = x2 - x1, y2 - y1
            gt_box_norm = ((x1 + x2) / 2 / 640, (y1 + y2) / 2 / 640, w / 640, h / 640)
            
            iou = calculate_iou(pred_box_norm, gt_box_norm)
            all_ious.append(iou)
            
            # Contar detecciones correctas para diferentes umbrales
            for threshold in [0.5, 0.75, 0.9]:
                if iou >= threshold:
                    num_correct_detections[threshold] += 1
            
            total_detections += 1
        
        # Mostrar progreso
        if (idx + 1) % 10 == 0:
            print(f"   Procesadas {idx + 1}/{len(dataset)} imágenes...", end='\r')
    
    print()  # Nueva línea después del progreso
    
    # Calcular métricas
    metrics = {
        'total_images': len(dataset),
        'total_detections': total_detections,
        'avg_iou': np.mean(all_ious) if all_ious else 0.0,
        'median_iou': np.median(all_ious) if all_ious else 0.0,
        'std_iou': np.std(all_ious) if all_ious else 0.0,
        'min_iou': np.min(all_ious) if all_ious else 0.0,
        'max_iou': np.max(all_ious) if all_ious else 0.0,
    }
    
    # Accuracy para diferentes umbrales
    for threshold in [0.5, 0.75, 0.9]:
        accuracy = num_correct_detections[threshold] / total_detections if total_detections > 0 else 0.0
        metrics[f'accuracy_iou_{threshold}'] = accuracy
    
    return metrics


def print_metrics(metrics: dict):
    """Imprimir métricas de forma legible."""
    print("\n" + "="*70)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*70)
    
    print(f"\n📊 Métricas Generales:")
    print(f"   • Total de imágenes: {metrics['total_images']}")
    print(f"   • Total de detecciones: {metrics['total_detections']}")
    
    print(f"\n📈 Métricas de IoU:")
    print(f"   • IoU Promedio: {metrics['avg_iou']:.4f} ({metrics['avg_iou']*100:.2f}%)")
    print(f"   • IoU Mediana: {metrics['median_iou']:.4f}")
    print(f"   • IoU Desviación: {metrics['std_iou']:.4f}")
    print(f"   • IoU Mínimo: {metrics['min_iou']:.4f}")
    print(f"   • IoU Máximo: {metrics['max_iou']:.4f}")
    
    print(f"\n🎯 Accuracy por Umbral:")
    for threshold in [0.5, 0.75, 0.9]:
        acc = metrics[f'accuracy_iou_{threshold}']
        print(f"   • IoU >= {threshold}: {acc:.4f} ({acc*100:.2f}%)")
    
    print("="*70)


def save_metrics(metrics: dict, output_path: str):
    """Guardar métricas a archivo JSON."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Métricas guardadas en: {output_path}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Evaluar modelo RetinaNet'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo .h5'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/retinanet_config.yaml',
        help='Ruta al archivo de configuración'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Ruta para guardar métricas (opcional)'
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Cargar modelo
    model = load_model(args.model)
    
    # Preparar dataset de test
    print("\n📂 Preparando dataset de evaluación...")
    full_dataset = RetinaNetDataset.from_pascal_voc(
        images_dir=config['data']['images_dir'],
        annotations_dir=config['data']['annotations_dir'],
        image_shape=tuple(config['data']['image_shape']),
        iou_threshold_pos=config['data']['iou_threshold_pos'],
        iou_threshold_neg=config['data']['iou_threshold_neg']
    )
    
    # Usar split de validación
    _, test_dataset = full_dataset.split(
        train_ratio=config['data']['train_ratio'],
        shuffle=False,
        seed=config['data']['seed']
    )
    
    print(f"✅ Dataset de test: {len(test_dataset)} imágenes")
    
    # Evaluar
    metrics = evaluate_model(model, test_dataset, config)
    
    # Mostrar resultados
    print_metrics(metrics)
    
    # Guardar métricas
    if args.output:
        save_metrics(metrics, args.output)
    else:
        # Guardar con nombre por defecto
        output_path = Path(config['paths']['results']) / f"{config['model']['name']}_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metrics(metrics, str(output_path))


if __name__ == '__main__':
    main()
