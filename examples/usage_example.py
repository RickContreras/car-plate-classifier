"""
Ejemplo de uso del Proyecto de Detección FC

Este script demuestra cómo usar todos los componentes del proyecto.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.models import FCNetwork
from src.data import DetectionDataset, normalize_bbox, denormalize_bbox
from src.training import Trainer, get_callbacks
from src.evaluation import MetricsCalculator


def example_feature_extraction():
    """Ejemplo: Extraer características de una imagen."""
    print("\n" + "="*60)
    print("Ejemplo 1: Extracción de Características")
    print("="*60)
    
    # Crear una imagen ficticia
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # Características HOG
    hog_extractor = HOGFeatureExtractor()
    hog_features = hog_extractor.extract(image)
    print(f"Forma de características HOG: {hog_features.shape}")
    print(f"Dimensión de características HOG: {hog_extractor.get_feature_dim()}")
    
    # Características BRISK
    brisk_extractor = BRISKFeatureExtractor(n_keypoints=512)
    brisk_features = brisk_extractor.extract(image)
    print(f"Forma de características BRISK: {brisk_features.shape}")
    print(f"Dimensión de características BRISK: {brisk_extractor.get_feature_dim()}")


def example_model_creation():
    """Ejemplo: Crear y compilar un modelo."""
    print("\n" + "="*60)
    print("Ejemplo 2: Creación de Modelo")
    print("="*60)
    
    # Crear modelo
    model = FCNetwork(
        input_dim=8100,
        architecture=[512, 256, 128, 64, 4],
        use_batch_norm=True
    )
    
    # Compilar
    model.compile(learning_rate=0.001)
    
    # Mostrar resumen
    model.summary()
    
    # Probar predicción
    X = np.random.randn(5, 8100).astype(np.float32)
    predictions = model.get_model().predict(X, verbose=0)
    print(f"\nForma de predicciones: {predictions.shape}")
    print(f"Predicción de muestra: {predictions[0]}")


def example_dataset_operations():
    """Ejemplo: Operaciones con dataset."""
    print("\n" + "="*60)
    print("Ejemplo 3: Operaciones con Dataset")
    print("="*60)
    
    # Crear datos ficticios
    n_samples = 100
    features = np.random.randn(n_samples, 8100).astype(np.float32)
    bboxes = np.random.rand(n_samples, 4).astype(np.float32)
    
    # Crear dataset
    dataset = DetectionDataset(features=features, bboxes=bboxes)
    print(f"Tamaño del dataset: {len(dataset)}")
    
    # Dividir dataset
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=42)
    print(f"Tamaño de entrenamiento: {len(train_ds)}")
    print(f"Tamaño de validación: {len(val_ds)}")
    
    # Obtener una muestra
    sample_features, sample_bbox = dataset[0]
    print(f"\nForma de características de muestra: {sample_features.shape}")
    print(f"Bbox de muestra: {sample_bbox}")


def example_bbox_utils():
    """Ejemplo: Utilidades de bounding box."""
    print("\n" + "="*60)
    print("Ejemplo 4: Utilidades de Bounding Box")
    print("="*60)
    
    # Bbox original en píxeles
    bbox_pixels = (100, 150, 300, 250)  # xmin, ymin, xmax, ymax
    img_width, img_height = 640, 480
    
    print(f"Bbox original (píxeles): {bbox_pixels}")
    print(f"Tamaño de imagen: {img_width}x{img_height}")
    
    # Normalizar
    bbox_norm = normalize_bbox(bbox_pixels, img_width, img_height)
    print(f"Bbox normalizado: {bbox_norm}")
    
    # Desnormalizar de vuelta
    bbox_recovered = denormalize_bbox(bbox_norm, img_width, img_height)
    print(f"Bbox recuperado (píxeles): {bbox_recovered}")


def example_metrics():
    """Ejemplo: Calcular métricas."""
    print("\n" + "="*60)
    print("Ejemplo 5: Cálculo de Métricas")
    print("="*60)
    
    # Crear predicciones y verdad real ficticias
    n_samples = 50
    y_true = np.random.rand(n_samples, 4).astype(np.float32)
    y_pred = y_true + np.random.randn(n_samples, 4).astype(np.float32) * 0.1
    y_pred = np.clip(y_pred, 0, 1)
    
    # Calcular métricas
    metrics_calc = MetricsCalculator(iou_thresholds=[0.5, 0.75, 0.9])
    metrics_calc.update(y_true, y_pred)
    metrics = metrics_calc.compute()
    
    # Imprimir métricas
    metrics_calc.print_metrics(metrics)


def example_training_setup():
    """Ejemplo: Configurar entrenamiento."""
    print("\n" + "="*60)
    print("Ejemplo 6: Configuración de Entrenamiento")
    print("="*60)
    
    # Crear datos ficticios
    X_train = np.random.randn(80, 8100).astype(np.float32)
    y_train = np.random.rand(80, 4).astype(np.float32)
    X_val = np.random.randn(20, 8100).astype(np.float32)
    y_val = np.random.rand(20, 4).astype(np.float32)
    
    # Crear modelo
    model = FCNetwork(input_dim=8100)
    model.compile(learning_rate=0.001)
    
    # Obtener callbacks
    callbacks = get_callbacks(
        model_name='example_model',
        patience=5,
        reduce_lr_patience=3
    )
    
    print(f"Callbacks: {[type(cb).__name__ for cb in callbacks]}")
    
    # Crear entrenador
    trainer = Trainer(
        model=model.get_model(),
        save_dir='models',
        name='example_model'
    )
    
    print("\nEntrenador creado exitosamente")
    print("Listo para entrenar con: trainer.train(X_train, y_train, X_val, y_val, ...)")


def main():
    """Ejecutar todos los ejemplos."""
    print("\n" + "="*70)
    print(" Proyecto de Detección FC - Ejemplos de Uso")
    print("="*70)
    
    # Ejecutar ejemplos
    example_feature_extraction()
    example_model_creation()
    example_dataset_operations()
    example_bbox_utils()
    example_metrics()
    example_training_setup()
    
    print("\n" + "="*70)
    print(" ¡Todos los ejemplos completados exitosamente!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
