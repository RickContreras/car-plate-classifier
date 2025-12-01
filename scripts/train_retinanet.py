#!/usr/bin/env python3
"""
Script para entrenar modelo RetinaNet para detección de placas vehiculares.

Este script sigue la misma estructura que train.py pero para RetinaNet.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
import json

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.retinanet import RetinaNetDetector
from src.data.retinanet_dataset import RetinaNetDataset
from tensorflow import keras


def load_config(config_path: str) -> dict:
    """
    Cargar configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con configuración
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_gpu(config: dict):
    """
    Configurar GPU según configuración.
    
    Args:
        config: Diccionario de configuración
    """
    if not config['hardware']['gpu']:
        # Desactivar GPU
        tf.config.set_visible_devices([], 'GPU')
        print("🚫 GPU desactivada, usando CPU")
    else:
        # Listar GPUs disponibles
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU(s) disponible(s): {len(gpus)}")
            # Configurar growth de memoria
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("⚠️  No se encontraron GPUs, usando CPU")
    
    # Mixed precision
    if config['hardware'].get('mixed_precision', False):
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision activado")


def create_model(config: dict) -> keras.Model:
    """
    Crear y compilar modelo RetinaNet.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Modelo compilado
    """
    print("\n" + "="*70)
    print("CREANDO MODELO RETINANET")
    print("="*70)
    
    # Crear detector
    detector = RetinaNetDetector(
        num_classes=config['model']['num_classes'],
        input_shape=tuple(config['model']['input_shape']),
        backbone_type=config['model']['backbone']['type'],
        backbone_weights=config['model']['backbone']['weights'],
        backbone_trainable=config['model']['backbone']['trainable'],
        feature_size=config['model']['fpn']['feature_size'],
        num_conv_layers=config['model']['subnets']['num_conv_layers'],
        anchor_sizes=config['model']['anchors']['sizes'],
        anchor_scales=config['model']['anchors']['scales'],
        anchor_ratios=config['model']['anchors']['aspect_ratios'],
        name=config['model']['name']
    )
    
    # Construir modelo
    model = detector.build()
    
    # Compilar modelo
    model = detector.compile_model(
        model,
        learning_rate=config['training']['learning_rate'],
        alpha=config['training']['loss']['focal_loss']['alpha'],
        gamma=config['training']['loss']['focal_loss']['gamma'],
        lambda_box=config['training']['loss']['lambda_box']
    )
    
    # Build model with input shape before counting params
    input_shape = tuple(config['model']['input_shape'])
    model.build(input_shape=(None, *input_shape))
    
    print(f"\n📊 Resumen del modelo:")
    print(f"   • Backbone: {config['model']['backbone']['type']}")
    print(f"   • Input shape: {config['model']['input_shape']}")
    print(f"   • Num anchors: {detector.num_anchors}")
    print(f"   • Parámetros totales: {model.count_params():,}")
    
    return model


def prepare_datasets(config: dict) -> tuple:
    """
    Preparar datasets de entrenamiento y validación.
    
    Args:
        config: Diccionario de configuración
        
    Returns:
        Tupla de (train_dataset, val_dataset, train_size, val_size)
    """
    print("\n" + "="*70)
    print("PREPARANDO DATASETS")
    print("="*70)
    
    # Crear dataset completo
    full_dataset = RetinaNetDataset.from_pascal_voc(
        images_dir=config['data']['images_dir'],
        annotations_dir=config['data']['annotations_dir'],
        image_shape=tuple(config['data']['image_shape']),
        iou_threshold_pos=config['data']['iou_threshold_pos'],
        iou_threshold_neg=config['data']['iou_threshold_neg'],
        augment=config['data']['augmentation']['enabled']
    )
    
    print(f"✅ Dataset cargado: {len(full_dataset)} imágenes")
    
    # Dividir en train/val
    train_dataset, val_dataset = full_dataset.split(
        train_ratio=config['data']['train_ratio'],
        shuffle=config['data']['shuffle'],
        seed=config['data']['seed']
    )
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    print(f"   • Train: {train_size} imágenes")
    print(f"   • Val: {val_size} imágenes")
    
    # Convertir a tf.data.Dataset
    train_tf_dataset = train_dataset.get_tf_dataset(
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_tf_dataset = val_dataset.get_tf_dataset(
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    return train_tf_dataset, val_tf_dataset, train_size, val_size


def create_callbacks(config: dict, model_name: str) -> list:
    """
    Crear callbacks para entrenamiento.
    
    Args:
        config: Diccionario de configuración
        model_name: Nombre del modelo
        
    Returns:
        Lista de callbacks
    """
    callbacks = []
    
    # Early Stopping
    if config['training']['callbacks']['early_stopping']['enabled']:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=config['training']['callbacks']['early_stopping']['monitor'],
            patience=config['training']['callbacks']['early_stopping']['patience'],
            mode=config['training']['callbacks']['early_stopping']['mode'],
            restore_best_weights=True,
            verbose=1
        ))
    
    # Reduce LR on Plateau
    if config['training']['callbacks']['reduce_lr']['enabled']:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=config['training']['callbacks']['reduce_lr']['monitor'],
            factor=config['training']['callbacks']['reduce_lr']['factor'],
            patience=config['training']['callbacks']['reduce_lr']['patience'],
            min_lr=config['training']['callbacks']['reduce_lr']['min_lr'],
            mode=config['training']['callbacks']['reduce_lr']['mode'],
            verbose=1
        ))
    
    # Model Checkpoint
    if config['training']['callbacks']['model_checkpoint']['enabled']:
        checkpoint_dir = Path(config['paths']['checkpoints'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Callback personalizado para guardar el modelo base sin wrapper
        class BaseModelCheckpoint(keras.callbacks.Callback):
            def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min', verbose=1):
                super().__init__()
                self.filepath = filepath
                self.monitor = monitor
                self.save_best_only = save_best_only
                self.mode = mode
                self.verbose = verbose
                self.best = np.inf if mode == 'min' else -np.inf
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                
                if current is None:
                    return
                
                # Verificar si mejoró
                improved = False
                if self.mode == 'min':
                    improved = current < self.best
                else:
                    improved = current > self.best
                
                if not self.save_best_only or improved:
                    if improved:
                        self.best = current
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch + 1}: {self.monitor} improved to {current:.5f}, saving model base to {self.filepath}")
                    
                    # Guardar el modelo base sin el wrapper
                    base_model = self.model.base_model
                    base_model.save(self.filepath, save_format='h5', include_optimizer=False)
        
        callbacks.append(BaseModelCheckpoint(
            filepath=str(checkpoint_dir / f'{model_name}_best.h5'),
            monitor=config['training']['callbacks']['model_checkpoint']['monitor'],
            save_best_only=config['training']['callbacks']['model_checkpoint']['save_best_only'],
            mode=config['training']['callbacks']['model_checkpoint']['mode'],
            verbose=1
        ))
    
    # TensorBoard
    if config['training']['callbacks']['tensorboard']['enabled']:
        log_dir = Path(config['paths']['logs']) / datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            update_freq=config['training']['callbacks']['tensorboard']['update_freq']
        ))
    
    return callbacks


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    train_size: int,
    val_size: int,
    config: dict
) -> keras.callbacks.History:
    """
    Entrenar modelo RetinaNet.
    
    Args:
        model: Modelo a entrenar
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validación
        train_size: Número de muestras de entrenamiento
        val_size: Número de muestras de validación
        config: Configuración
        
    Returns:
        Historial de entrenamiento
    """
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO")
    print("="*70)
    
    # Calcular steps
    batch_size = config['training']['batch_size']
    steps_per_epoch = train_size // batch_size
    validation_steps = val_size // batch_size
    
    print(f"   • Steps por epoch: {steps_per_epoch}")
    print(f"   • Validation steps: {validation_steps}")
    
    # Crear callbacks
    callbacks = create_callbacks(config, config['model']['name'])
    
    # Entrenar
    history = model.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_results(model: keras.Model, history: keras.callbacks.History, config: dict):
    """
    Guardar modelo y resultados.
    
    Args:
        model: Modelo entrenado
        history: Historial de entrenamiento
        config: Configuración
    """
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    # Crear directorios
    models_dir = Path(config['paths']['models'])
    results_dir = Path(config['paths']['results'])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = config['model']['name']
    
    # Guardar modelo
    model_path = models_dir / f'{model_name}.h5'
    model.save(model_path)
    print(f"✅ Modelo guardado: {model_path}")
    
    # Guardar historial
    history_path = results_dir / f'{model_name}_history.json'
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=2)
    print(f"✅ Historial guardado: {history_path}")
    
    # Guardar configuración
    config_path = results_dir / f'{model_name}_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Configuración guardada: {config_path}")
    
    print("\n🎉 Entrenamiento completado exitosamente!")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description='Entrenar modelo RetinaNet para detección de placas'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/retinanet_config.yaml',
        help='Ruta al archivo de configuración YAML'
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    print("📄 Cargando configuración...")
    config = load_config(args.config)
    
    # Configurar GPU
    setup_gpu(config)
    
    # Establecer semillas para reproducibilidad
    if config['reproducibility']['deterministic']:
        np.random.seed(config['reproducibility']['seed'])
        tf.random.set_seed(config['reproducibility']['seed'])
    
    # Crear modelo
    model = create_model(config)
    
    # Preparar datasets
    train_dataset, val_dataset, train_size, val_size = prepare_datasets(config)
    
    # Entrenar
    history = train_model(model, train_dataset, val_dataset, train_size, val_size, config)
    
    # Guardar resultados
    save_results(model, history, config)


if __name__ == '__main__':
    main()
