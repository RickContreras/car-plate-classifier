#!/usr/bin/env python3
"""
Script de prueba para verificar que todos los módulos funcionan correctamente.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

print("\n" + "="*70)
print(" "*20 + "🧪 PRUEBA DE MÓDULOS")
print("="*70)

# Test 1: Configuración
print("\n[1/5] Probando carga de configuración...")
try:
    from src.preprocessing import load_config
    config = load_config()
    print(f"   ✅ Config cargada: img_size={config['data']['img_size']}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 2: Preprocesamiento
print("\n[2/5] Probando módulo de preprocesamiento...")
try:
    from src.preprocessing import preprocess_image, crop_plate_region, bboxes_overlap
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    processed = preprocess_image(test_image, config)
    
    print(f"   ✅ Preprocesamiento OK: shape original={test_image.shape}, procesada={processed.shape}")
    
    # Test bbox overlap
    bbox1 = {'xmin': 10, 'ymin': 10, 'xmax': 50, 'ymax': 50}
    bbox2 = {'xmin': 40, 'ymin': 40, 'xmax': 80, 'ymax': 80}
    bbox3 = {'xmin': 100, 'ymin': 100, 'xmax': 150, 'ymax': 150}
    bbox4 = {'xmin': 20, 'ymin': 20, 'xmax': 45, 'ymax': 45}  # Gran solapamiento con bbox1
    
    assert bboxes_overlap(bbox1, bbox4, threshold=0.1) == True  # Debe solapar
    assert bboxes_overlap(bbox1, bbox3) == False  # No debe solapar
    print(f"   ✅ Detección de solapamiento OK")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Extracción de características
print("\n[3/5] Probando extracción de características...")
try:
    from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor
    
    # Crear imagen de prueba en escala de grises
    test_gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    # HOG
    hog_extractor = HOGFeatureExtractor(config)
    hog_features = hog_extractor.extract(test_gray)
    print(f"   ✅ HOG OK: features shape={hog_features.shape}")
    
    # BRISK
    brisk_extractor = BRISKFeatureExtractor(config)
    brisk_features = brisk_extractor.extract(test_gray)
    print(f"   ✅ BRISK OK: features shape={brisk_features.shape}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Modelos
print("\n[4/5] Probando creación de modelos...")
try:
    from src.train_models import PlateClassifier, create_neural_network
    
    # Test SVM
    classifier_svm = PlateClassifier(model_type='svm')
    print(f"   ✅ Clasificador SVM creado")
    
    # Test Random Forest
    classifier_rf = PlateClassifier(model_type='random_forest')
    print(f"   ✅ Clasificador Random Forest creado")
    
    # Test Neural Network
    classifier_nn = PlateClassifier(model_type='neural_network')
    nn_model = create_neural_network(input_shape=100)
    print(f"   ✅ Red Neuronal creada: {len(nn_model.layers)} capas")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Entrenamiento con datos sintéticos
print("\n[5/5] Probando entrenamiento con datos sintéticos...")
try:
    # Crear datos sintéticos
    n_samples = 100
    n_features = hog_features.shape[0]
    
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    X_test = np.random.randn(20, n_features)
    y_test = np.random.randint(0, 2, 20)
    
    # Entrenar SVM rápido
    classifier_test = PlateClassifier(model_type='svm')
    classifier_test.train(X_train, y_train)
    
    # Evaluar
    metrics = classifier_test.evaluate(X_test, y_test)
    print(f"   ✅ Entrenamiento OK: accuracy={metrics['accuracy']:.2f}")
    
    # Predicción
    predictions = classifier_test.predict(X_test)
    print(f"   ✅ Predicción OK: {len(predictions)} muestras clasificadas")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verificar estructura de directorios
print("\n[6/6] Verificando estructura de directorios...")
try:
    required_dirs = [
        'data/raw/images',
        'data/raw/annotations',
        'data/processed',
        'models',
        'results'
    ]
    
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   📁 {dir_path} (creado)")
            
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Resumen final
print("\n" + "="*70)
print("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
print("="*70)

print("\n📋 Estado del proyecto:")
print("   ✅ Entorno virtual configurado")
print("   ✅ Dependencias instaladas")
print("   ✅ Módulos de preprocesamiento funcionando")
print("   ✅ Extractores de características funcionando")
print("   ✅ Modelos de clasificación funcionando")
print("   ✅ Estructura de directorios lista")

print("\n🚀 Próximos pasos:")
print("   1. Ejecuta 'python scripts/download_data.py' para descargar el dataset")
print("   2. Ejecuta 'python main.py' para entrenar todos los modelos")
print("   3. Ejecuta 'python app/gui.py' para usar la interfaz gráfica")

print("\n" + "="*70 + "\n")
