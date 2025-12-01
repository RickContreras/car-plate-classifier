#!/usr/bin/env python3
"""
Test rápido de implementación de RetinaNet.

Verifica que todos los componentes funcionen correctamente.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Agregar path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("🧪 TEST RÁPIDO: RETINANET IMPLEMENTATION")
print("="*70)

# Test 1: Imports
print("\n1️⃣  Verificando imports...")
try:
    from src.models.retinanet import (
        RetinaNetDetector,
        AnchorGenerator,
        FocalLoss,
        SmoothL1Loss,
        ResNetBackbone,
        FeaturePyramidNetwork
    )
    print("   ✅ Todos los módulos importados correctamente")
except Exception as e:
    print(f"   ❌ Error en imports: {e}")
    sys.exit(1)

# Test 2: Crear AnchorGenerator
print("\n2️⃣  Probando AnchorGenerator...")
try:
    anchor_gen = AnchorGenerator(
        sizes=[32, 64, 128, 256, 512],
        scales=[1.0, 1.26, 1.59],
        aspect_ratios=[0.5, 1.0, 2.0]
    )
    anchors = anchor_gen.generate_anchors((640, 640))
    print(f"   ✅ Anchors generadas: {anchors.shape}")
    print(f"   ℹ️  Número de anchors: {anchor_gen.num_anchors} por posición")
except Exception as e:
    print(f"   ❌ Error en AnchorGenerator: {e}")
    sys.exit(1)

# Test 3: Crear Losses
print("\n3️⃣  Probando funciones de pérdida...")
try:
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0, from_logits=True)
    smooth_l1 = SmoothL1Loss(delta=1.0)
    print(f"   ✅ FocalLoss creada: {focal_loss}")
    print(f"   ✅ SmoothL1Loss creada: {smooth_l1}")
except Exception as e:
    print(f"   ❌ Error en losses: {e}")
    sys.exit(1)

# Test 4: Crear Backbone
print("\n4️⃣  Probando ResNet Backbone...")
try:
    backbone = ResNetBackbone(
        input_shape=(640, 640, 3),
        weights='imagenet',
        trainable=False
    )
    backbone_model = backbone.build()
    print(f"   ✅ Backbone creado: {backbone_model.name}")
    print(f"   ℹ️  Outputs: {len(backbone_model.outputs)} niveles (C2-C5)")
except Exception as e:
    print(f"   ❌ Error en Backbone: {e}")
    sys.exit(1)

# Test 5: Crear FPN
print("\n5️⃣  Probando Feature Pyramid Network...")
try:
    fpn = FeaturePyramidNetwork(feature_size=256, num_levels=5)
    print(f"   ✅ FPN creado: {fpn}")
    print(f"   ℹ️  Feature size: {fpn.feature_size} canales")
except Exception as e:
    print(f"   ❌ Error en FPN: {e}")
    sys.exit(1)

# Test 6: Crear Detector Completo
print("\n6️⃣  Probando RetinaNet Detector...")
try:
    detector = RetinaNetDetector(
        num_classes=1,
        input_shape=(640, 640, 3),
        backbone_type='resnet50',
        backbone_weights='imagenet',
        backbone_trainable=False,
        feature_size=256,
        num_conv_layers=4
    )
    print(f"   ✅ Detector creado: {detector}")
    print(f"   ℹ️  Num classes: {detector.num_classes}")
    print(f"   ℹ️  Num anchors: {detector.num_anchors} por posición")
except Exception as e:
    print(f"   ❌ Error en Detector: {e}")
    sys.exit(1)

# Test 7: Construir Modelo
print("\n7️⃣  Construyendo modelo completo...")
try:
    model = detector.build()
    print(f"   ✅ Modelo construido: {model.name}")
    print(f"   ℹ️  Inputs: {model.input_shape}")
    print(f"   ℹ️  Outputs: {len(model.outputs)} (clasificación + regresión)")
    print(f"   ℹ️  Parámetros totales: {model.count_params():,}")
except Exception as e:
    print(f"   ❌ Error construyendo modelo: {e}")
    sys.exit(1)

# Test 8: Forward Pass
print("\n8️⃣  Probando forward pass...")
try:
    # Crear imagen dummy
    dummy_image = np.random.rand(1, 640, 640, 3).astype(np.float32)
    
    # Predecir
    cls_pred, box_pred = model.predict(dummy_image, verbose=0)
    
    print(f"   ✅ Forward pass exitoso")
    print(f"   ℹ️  Classification output: {cls_pred.shape}")
    print(f"   ℹ️  Box regression output: {box_pred.shape}")
except Exception as e:
    print(f"   ❌ Error en forward pass: {e}")
    sys.exit(1)

# Test 9: Compilar Modelo
print("\n9️⃣  Compilando modelo...")
try:
    model = detector.compile_model(
        model,
        learning_rate=1e-4,
        alpha=0.25,
        gamma=2.0,
        lambda_box=1.0
    )
    print(f"   ✅ Modelo compilado exitosamente")
    print(f"   ℹ️  Optimizer: {model.optimizer.__class__.__name__}")
except Exception as e:
    print(f"   ❌ Error compilando modelo: {e}")
    sys.exit(1)

# Test 10: Verificar Dataset
print("\n🔟 Verificando RetinaNetDataset...")
try:
    from src.data.retinanet_dataset import RetinaNetDataset
    
    # Verificar que existe directorio de datos
    data_exists = Path('data/raw/images').exists() and Path('data/raw/annotations').exists()
    
    if data_exists:
        print(f"   ✅ RetinaNetDataset importado")
        print(f"   ℹ️  Datos encontrados en data/raw/")
    else:
        print(f"   ⚠️  RetinaNetDataset importado pero datos no encontrados")
        print(f"   ℹ️  Ejecuta: python scripts/download_data.py")
        
except Exception as e:
    print(f"   ❌ Error en Dataset: {e}")
    sys.exit(1)

# Resumen Final
print("\n" + "="*70)
print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
print("="*70)

print("\n📊 Resumen de Implementación:")
print(f"   • AnchorGenerator: ✅")
print(f"   • FocalLoss + SmoothL1Loss: ✅")
print(f"   • ResNetBackbone: ✅")
print(f"   • FeaturePyramidNetwork: ✅")
print(f"   • RetinaNetDetector: ✅")
print(f"   • RetinaNetDataset: ✅")
print(f"   • Forward Pass: ✅")
print(f"   • Compilación: ✅")

print(f"\n💡 Modelo listo con {model.count_params():,} parámetros")
print(f"   • Input: (640, 640, 3)")
print(f"   • Output: ~{cls_pred.shape[1]:,} anchors")
print(f"   • Backbone: ResNet50 (ImageNet)")
print(f"   • FPN: 256 canales")

print("\n🚀 Próximos pasos:")
print("   1. Entrenar: python scripts/train_retinanet.py")
print("   2. Evaluar: python scripts/evaluate_retinanet.py --model models/retinanet_plates.h5")
print("   3. Ejemplo: python examples/retinanet_example.py")

print("\n" + "="*70)
print("🎉 ¡RetinaNet implementado correctamente!")
print("="*70)
