#!/usr/bin/env python3
"""
Script para evaluar modelos de detección entrenados.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import json
import os

# Desactivar GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras
from src.data import load_dataset
from src.evaluation import MetricsCalculator


def load_config(config_path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model_path: str, dataset_path: str, config: dict, feature_type: str):
    """
    Evaluar un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo entrenado
        dataset_path: Ruta al dataset
        config: Diccionario de configuración
        feature_type: 'hog' o 'brisk'
    """
    print(f"\n{'='*60}")
    print(f"Evaluando Modelo de Detección {feature_type.upper()}")
    print(f"{'='*60}\n")
    
    # Cargar modelo
    print(f"Cargando modelo desde: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("  Modelo cargado")
    
    # Cargar dataset
    print(f"\nCargando dataset desde: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"  Cargadas {len(dataset)} muestras")
    
    # Dividir dataset para obtener conjunto de prueba
    data_config = config['data']
    train_dataset, test_dataset = dataset.split(
        train_ratio=data_config['train_ratio'],
        shuffle=data_config['shuffle'],
        seed=data_config.get('seed', 42)
    )
    
    X_test = test_dataset.features
    y_test = test_dataset.bboxes
    
    print(f"  Muestras de prueba: {len(test_dataset)}")
    
    # Realizar predicciones
    print("\nRealizando predicciones...")
    y_pred = model.predict(X_test, verbose=0)
    print("  Predicciones completadas")
    
    # Calcular métricas
    print("\nCalculando métricas...")
    metrics_calc = MetricsCalculator(
        iou_thresholds=config['evaluation']['iou_thresholds']
    )
    metrics_calc.update(y_test, y_pred)
    metrics = metrics_calc.compute()
    
    # Imprimir métricas
    metrics_calc.print_metrics(metrics)
    
    # Guardar métricas a archivo
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    metrics_path = results_dir / f"{model_name}_eval_metrics.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"  Métricas guardadas en: {metrics_path}")
    
    # Imprimir predicciones de muestra
    print("\nPredicciones de Muestra:")
    print("-" * 60)
    n_samples = min(5, len(y_test))
    for i in range(n_samples):
        true = y_test[i]
        pred = y_pred[i]
        iou = metrics_calc._calculate_iou(pred, true)
        
        print(f"\nMuestra {i+1}:")
        print(f"  Verdad Real: [{true[0]:.3f}, {true[1]:.3f}, {true[2]:.3f}, {true[3]:.3f}]")
        print(f"  Predicción:  [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}, {pred[3]:.3f}]")
        print(f"  IoU: {iou:.3f}")
    print("-" * 60)
    
    print(f"\n{'='*60}")
    print("¡Evaluación completada!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluar modelo de detección')
    parser.add_argument('--model', type=str, required=True, help='Ruta del archivo de modelo')
    parser.add_argument('--dataset', type=str, help='Ruta del archivo de dataset')
    parser.add_argument('--config', type=str, help='Ruta del archivo de configuración')
    parser.add_argument('--feature-type', type=str, choices=['hog', 'brisk'], help='Tipo de características')
    
    args = parser.parse_args()
    
    # Determinar tipo de características
    if args.feature_type:
        feature_type = args.feature_type
    else:
        # Extraer del nombre del modelo
        model_name = Path(args.model).stem
        if 'hog' in model_name:
            feature_type = 'hog'
        elif 'brisk' in model_name:
            feature_type = 'brisk'
        else:
            print("ERROR: No se puede determinar el tipo de características. Use --feature-type")
            return
    
    # Cargar configuración
    if args.config:
        config = load_config(args.config)
    else:
        config_file = f"configs/{feature_type}_config.yaml"
        if Path(config_file).exists():
            config = load_config(config_file)
        else:
            print(f"ERROR: Archivo de configuración no encontrado: {config_file}")
            return
    
    # Establecer ruta del dataset
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = f"data/processed/detection_{feature_type}.pkl"
    
    # Verificar que los archivos existan
    if not Path(args.model).exists():
        print(f"ERROR: Modelo no encontrado: {args.model}")
        return
    
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset no encontrado: {dataset_path}")
        return
    
    # Evaluar modelo
    evaluate_model(args.model, dataset_path, config, feature_type)


if __name__ == '__main__':
    main()
