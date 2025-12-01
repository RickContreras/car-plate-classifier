#!/usr/bin/env python3
"""
Ejemplo rápido de uso de RetinaNet para detección de placas.

Este script muestra cómo usar el modelo RetinaNet de forma simple.
"""

import sys
from pathlib import Path
import tensorflow as tf
import cv2
import numpy as np

# Agregar path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.retinanet import RetinaNetDetector
from src.data.retinanet_dataset import RetinaNetDataset


def ejemplo_basico():
    """Ejemplo básico: crear y entrenar modelo."""
    
    print("="*70)
    print("EJEMPLO 1: Crear y Entrenar RetinaNet")
    print("="*70)
    
    # 1. Crear detector
    print("\n1️⃣  Creando detector...")
    detector = RetinaNetDetector(
        num_classes=1,
        input_shape=(640, 640, 3),
        backbone_type='resnet50',
        backbone_weights='imagenet'
    )
    
    # 2. Construir modelo
    print("2️⃣  Construyendo modelo...")
    model = detector.build()
    
    # 3. Compilar
    print("3️⃣  Compilando modelo...")
    model = detector.compile_model(model, learning_rate=1e-4)
    
    print(f"\n✅ Modelo listo con {model.count_params():,} parámetros")
    
    # 4. Cargar dataset
    print("\n4️⃣  Cargando dataset...")
    dataset = RetinaNetDataset.from_pascal_voc(
        images_dir='data/raw/images',
        annotations_dir='data/raw/annotations',
        image_shape=(640, 640)
    )
    
    print(f"✅ Dataset cargado: {len(dataset)} imágenes")
    
    # 5. Split train/val
    print("\n5️⃣  Dividiendo dataset...")
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=42)
    
    print(f"   • Train: {len(train_ds)} imágenes")
    print(f"   • Val: {len(val_ds)} imágenes")
    
    # 6. Entrenar (ejemplo corto)
    print("\n6️⃣  Entrenando modelo (2 epochs de ejemplo)...")
    history = model.fit(
        train_ds.get_tf_dataset(batch_size=4),
        validation_data=val_ds.get_tf_dataset(batch_size=4),
        epochs=2,
        verbose=1
    )
    
    print("\n✅ Entrenamiento completado!")
    
    return model


def ejemplo_inferencia(model):
    """Ejemplo de inferencia en una imagen."""
    
    print("\n" + "="*70)
    print("EJEMPLO 2: Inferencia en Nueva Imagen")
    print("="*70)
    
    # Cargar imagen de ejemplo
    image_path = 'data/raw/images/Cars0.png'
    
    if not Path(image_path).exists():
        print(f"⚠️  Imagen no encontrada: {image_path}")
        return
    
    print(f"\n📷 Cargando imagen: {image_path}")
    
    # Preprocesar imagen
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # Expandir dimensión de batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predecir
    print("🔮 Realizando predicción...")
    cls_pred, box_pred = model.predict(img_batch, verbose=0)
    
    # Aplicar sigmoid
    cls_pred = tf.nn.sigmoid(cls_pred[0]).numpy()
    
    # Obtener top 5 detecciones
    scores = cls_pred[:, 0]
    top_indices = np.argsort(scores)[::-1][:5]
    
    print(f"\n📊 Top 5 Detecciones:")
    for i, idx in enumerate(top_indices, 1):
        score = scores[idx]
        print(f"   {i}. Score: {score:.4f}")
    
    print("\n✅ Inferencia completada!")


def ejemplo_comparacion():
    """Ejemplo de comparación HOG vs RetinaNet."""
    
    print("\n" + "="*70)
    print("EJEMPLO 3: Comparación de Arquitecturas")
    print("="*70)
    
    print("\n📊 Comparación HOG + FC vs RetinaNet:\n")
    
    print("┌─────────────────┬──────────────┬──────────────┐")
    print("│ Característica  │ HOG + FC     │ RetinaNet    │")
    print("├─────────────────┼──────────────┼──────────────┤")
    print("│ Tipo            │ 2-Stage      │ End-to-End   │")
    print("│ Features        │ Manuales     │ Aprendidas   │")
    print("│ Parámetros      │ 4.3M         │ 23M          │")
    print("│ IoU Esperado    │ ~40%         │ ~65%         │")
    print("│ Velocidad       │ 10 FPS       │ 20 FPS       │")
    print("│ Memoria         │ Baja         │ Alta         │")
    print("│ Training Time   │ 30 min       │ 2-3 horas    │")
    print("└─────────────────┴──────────────┴──────────────┘")
    
    print("\n💡 Recomendación:")
    print("   • Producción/Precisión → RetinaNet")
    print("   • Prototipado Rápido → HOG + FC")
    print("   • Dispositivos IoT → BRISK + FC")


def main():
    """Función principal con menú interactivo."""
    
    print("\n" + "="*70)
    print("🚗 EJEMPLOS DE USO: Car Plate Classifier")
    print("="*70)
    
    print("\nSelecciona un ejemplo:")
    print("  1. Crear y entrenar RetinaNet")
    print("  2. Inferencia en imagen (requiere modelo entrenado)")
    print("  3. Comparación de arquitecturas")
    print("  4. Ejecutar todos los ejemplos")
    print("  0. Salir")
    
    choice = input("\n👉 Opción: ").strip()
    
    if choice == '1':
        ejemplo_basico()
    
    elif choice == '2':
        print("\n⚠️  Este ejemplo requiere un modelo pre-entrenado.")
        print("   Entrena un modelo primero con la opción 1.")
        
    elif choice == '3':
        ejemplo_comparacion()
    
    elif choice == '4':
        modelo = ejemplo_basico()
        ejemplo_inferencia(modelo)
        ejemplo_comparacion()
    
    elif choice == '0':
        print("\n👋 ¡Hasta luego!")
    
    else:
        print("\n❌ Opción inválida")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
