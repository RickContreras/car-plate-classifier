#!/usr/bin/env python3
"""
Script para entrenar modelos de detección.

Entrena una red neuronal completamente conectada para regresión de bounding box.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os

# Desactivar GPU para evitar errores de CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.models import FCNetwork
from src.training import Trainer, get_callbacks
from src.training.callbacks import IoUCallback
from src.evaluation import MetricsCalculator


def load_config(config_path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_training_history(history, save_path: str):
    """Graficar y guardar historial de entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graficar pérdida
    axes[0].plot(history.history['loss'], label='Pérdida Entrenamiento', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Pérdida Validación', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Pérdida', fontsize=12)
    axes[0].set_title('Pérdida de Entrenamiento y Validación', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Graficar MAE
    axes[1].plot(history.history['mae'], label='MAE Entrenamiento', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='MAE Validación', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('MAE de Entrenamiento y Validación', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico de historial de entrenamiento guardado en: {save_path}")
    plt.close()


def train_model(config_path: str, dataset_path: str, feature_type: str):
    """
    Entrenar modelo de detección.
    
    Args:
        config_path: Ruta al archivo de configuración
        dataset_path: Ruta al archivo de dataset
        feature_type: 'hog' o 'brisk'
    """
    # Cargar configuración
    config = load_config(config_path)
    
    print(f"\n{'='*60}")
    print(f"Entrenando Modelo de Detección {feature_type.upper()}")
    print(f"{'='*60}\n")
    
    # Cargar dataset
    print("Cargando dataset...")
    dataset = load_dataset(dataset_path)
    print(f"✓ Cargadas {len(dataset)} muestras")
    
    # Dividir dataset
    data_config = config['data']
    train_dataset, val_dataset = dataset.split(
        train_ratio=data_config['train_ratio'],
        shuffle=data_config['shuffle'],
        seed=data_config.get('seed', 42)
    )
    
    X_train, y_train = train_dataset.features, train_dataset.bboxes
    X_val, y_val = val_dataset.features, val_dataset.bboxes
    
    print(f"✓ Muestras de entrenamiento: {len(train_dataset)}")
    print(f"✓ Muestras de validación: {len(val_dataset)}")
    print(f"✓ Dimensión de características: {X_train.shape[1]}")
    
    # Crear modelo
    model_config = config['model']
    architecture = [layer['units'] for layer in model_config['architecture']]
    activations = [layer['activation'] for layer in model_config['architecture']]
    dropout_rates = [layer.get('dropout', 0.0) for layer in model_config['architecture']]
    
    use_batch_norm = any(layer.get('batch_norm', False) for layer in model_config['architecture'][:-1])
    
    print("\nConstruyendo modelo...")
    fc_model = FCNetwork(
        input_dim=X_train.shape[1],
        architecture=architecture,
        activations=activations,
        use_batch_norm=use_batch_norm,
        dropout_rates=dropout_rates,
        l2_reg=model_config.get('l2_reg', 0.0),
        name=model_config['name']
    )
    
    # Compilar modelo
    training_config = config['training']
    fc_model.compile(
        optimizer=training_config['optimizer'],
        learning_rate=training_config['learning_rate'],
        loss=training_config['loss'],
        metrics=training_config['metrics']
    )
    
    print("\nResumen del Modelo:")
    fc_model.summary()
    
    # Configurar callbacks
    paths = config['paths']
    callbacks = get_callbacks(
        model_name=model_config['name'],
        save_dir=paths['models'],
        patience=training_config['callbacks']['early_stopping']['patience'],
        reduce_lr_patience=training_config['callbacks']['reduce_lr']['patience'],
        min_lr=training_config['callbacks']['reduce_lr']['min_lr'],
        monitor=training_config['callbacks']['early_stopping']['monitor'],
        mode=training_config['callbacks']['early_stopping']['mode']
    )
    
    # Agregar callback de IoU
    iou_callback = IoUCallback(validation_data=(X_val, y_val))
    callbacks.append(iou_callback)
    
    # Crear entrenador
    trainer = Trainer(
        model=fc_model.get_model(),
        save_dir=paths['models'],
        log_dir=paths['logs'],
        name=model_config['name']
    )
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...\n")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar modelo
    model_path = Path(paths['models']) / f"{model_config['name']}.h5"
    trainer.save_model(str(model_path))
    
    # Guardar historial
    trainer.save_history()
    
    # Graficar historial de entrenamiento
    results_dir = Path(paths['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / f"{model_config['name']}_training.png"
    plot_training_history(history, str(plot_path))
    
    # Evaluar en conjunto de validación
    print("\nEvaluando en conjunto de validación...")
    y_pred = trainer.predict(X_val)
    
    metrics_calc = MetricsCalculator(
        iou_thresholds=config['evaluation']['iou_thresholds']
    )
    metrics_calc.update(y_val, y_pred)
    metrics = metrics_calc.compute()
    metrics_calc.print_metrics(metrics)
    
    # Guardar métricas
    import json
    metrics_path = results_dir / f"{model_config['name']}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Métricas guardadas en: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("¡Entrenamiento completado exitosamente!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo de detección')
    parser.add_argument('--config', type=str, required=True, help='Ruta del archivo de configuración')
    parser.add_argument('--dataset', type=str, help='Ruta del archivo de dataset')
    parser.add_argument('--feature-type', type=str, choices=['hog', 'brisk'], help='Tipo de características')
    
    args = parser.parse_args()
    
    # Determinar tipo de características desde config o argumento
    if args.feature_type:
        feature_type = args.feature_type
    else:
        # Extraer del nombre del archivo de configuración
        config_name = Path(args.config).stem
        if 'hog' in config_name:
            feature_type = 'hog'
        elif 'brisk' in config_name:
            feature_type = 'brisk'
        else:
            print("ERROR: No se puede determinar el tipo de características. Use --feature-type")
            return
    
    # Establecer ruta del dataset
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = f"data/processed/detection_{feature_type}.pkl"
    
    # Verificar si el dataset existe
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset no encontrado: {dataset_path}")
        print("Ejecute prepare_dataset.py primero")
        return
    
    # Entrenar modelo
    train_model(args.config, dataset_path, feature_type)


if __name__ == '__main__':
    main()
