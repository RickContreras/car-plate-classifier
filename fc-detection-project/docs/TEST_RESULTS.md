# Resultados de Prueba del Proyecto

## Fecha: 26 de Octubre, 2025

## Resumen

Este documento resume las pruebas realizadas en el proyecto de detección de placas vehiculares utilizando redes neuronales totalmente conectadas (FC) con características HOG.

## Configuración del Entorno

### 1. Entorno Virtual
- Se creó un entorno virtual con Python 3.12
- Se instalaron todas las dependencias del archivo `requirements.txt`
- Se agregó `kaggle>=1.5.16` para la descarga de datos

### 2. Dataset
- **Fuente**: Kaggle - Car Plate Detection Dataset
- **URL**: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- **Total de imágenes**: 433
- **Tamaño del dataset**: 203.02 MB
- **Formato de anotaciones**: Pascal VOC XML
- **Estructura**:
  - `data/raw/images/`: 433 imágenes PNG
  - `data/raw/annotations/`: 433 archivos XML con bounding boxes

## Preparación del Dataset

### Características HOG Extraídas
```
Feature extractor: HOGFeatureExtractor(name='HOG', dim=42849)
Processed: 433 samples
Features shape: (433, 42849)
Bboxes shape: (433, 4)
```

### Características BRISK Extraídas
```
Feature extractor: BRISKFeatureExtractor(name='BRISK', dim=32768)
Processed: 433 samples
Features shape: (433, 32768)
Bboxes shape: (433, 4)
```

## Entrenamiento del Modelo

### Arquitectura de los Modelos

#### Modelo HOG
```
Total params: 22,115,780 (84.37 MB)
Trainable params: 22,113,860 (84.36 MB)
Non-trainable params: 1,920 (7.50 KB)

Capas:
- Input: 42849 características HOG
- Dense 1: 512 unidades + BatchNorm + ReLU + Dropout(0.4)
- Dense 2: 256 unidades + BatchNorm + ReLU + Dropout(0.3)
- Dense 3: 128 unidades + BatchNorm + ReLU + Dropout(0.2)
- Dense 4: 64 unidades + BatchNorm + ReLU + Dropout(0.1)
- Output: 4 unidades (bbox) + Sigmoid
```

#### Modelo BRISK
```
Total params: 16,954,308 (64.68 MB)
Trainable params: 16,952,388 (64.67 MB)
Non-trainable params: 1,920 (7.50 KB)

Capas:
- Input: 32768 características BRISK
- Dense 1: 512 unidades + BatchNorm + ReLU + Dropout(0.4)
- Dense 2: 256 unidades + BatchNorm + ReLU + Dropout(0.3)
- Dense 3: 128 unidades + BatchNorm + ReLU + Dropout(0.2)
- Dense 4: 64 unidades + BatchNorm + ReLU + Dropout(0.1)
- Output: 4 unidades (bbox) + Sigmoid
```

### Configuración de Entrenamiento
- **Train samples**: 346
- **Val samples**: 87
- **Épocas totales**: 75 (con early stopping)
- **Batch size**: 32
- **Optimizador**: Adam (lr inicial: 0.001)
- **Loss**: Mean Squared Error (MSE)
- **Callbacks**: 
  - Early Stopping (patience: 15)
  - ReduceLROnPlateau (patience: 5)
  - ModelCheckpoint (mejor modelo)
  - IoU Callback

### Resultados del Entrenamiento

#### Modelo HOG
**Mejor Época (60)**
```
val_loss: 0.00838
val_mae: 0.0555
avg_iou: 0.2998
learning_rate: 0.00025
Épocas totales: 75
```

**Métricas Finales en Validación**
```
Detection Metrics
============================================================
Total samples: 87

IoU Statistics:
  Average IoU: 0.2998
  Median IoU:  0.2561
  Std IoU:     0.2676
  Min IoU:     0.0000
  Max IoU:     0.7818

MAE: 0.0555

Accuracy at different IoU thresholds:
  IoU >= 0.5: 31.03% (27/87)
  IoU >= 0.75: 4.60% (4/87)
  IoU >= 0.9: 0.00% (0/87)
```

#### Modelo BRISK
**Mejor Época (82)**
```
val_loss: 0.01171
val_mae: 0.0667
avg_iou: 0.2457
learning_rate: 0.000125
Épocas totales: 97
```

**Métricas Finales en Validación**
```
Detection Metrics
============================================================
Total samples: 87

IoU Statistics:
  Average IoU: 0.2457
  Median IoU:  0.1080
  Std IoU:     0.2748
  Min IoU:     0.0000
  Max IoU:     0.9011

MAE: 0.0667

Accuracy at different IoU thresholds:
  IoU >= 0.5: 20.69% (18/87)
  IoU >= 0.75: 6.90% (6/87)
  IoU >= 0.9: 1.15% (1/87)
```

## Inferencia

### Pruebas Realizadas
Se realizaron pruebas de inferencia con 5 imágenes del dataset:

1. **Cars330.png**: Predicción (37, 217) - (49, 221)
2. **Cars92.png**: Predicción (288, 187) - (407, 253)
3. **Cars202.png**: Predicción (207, 134) - (273, 155)
4. **Cars284.png**: Predicción (75, 205) - (270, 257)
5. **Cars217.png**: Predicción (140, 191) - (216, 217)

### Archivos Generados
- `models/detection_hog_best.h5` (254 MB) - Mejor modelo HOG
- `models/detection_hog.h5` (254 MB) - Modelo final HOG
- `models/detection_hog_history.json` - Historia de entrenamiento HOG
- `models/detection_brisk_best.h5` (194 MB) - Mejor modelo BRISK
- `models/detection_brisk.h5` (194 MB) - Modelo final BRISK
- `models/detection_brisk_history.json` - Historia de entrenamiento BRISK
- `results/detection_hog_training.png` - Gráficos de entrenamiento HOG
- `results/detection_hog_metrics.json` - Métricas de evaluación HOG
- `results/detection_brisk_training.png` - Gráficos de entrenamiento BRISK
- `results/detection_brisk_metrics.json` - Métricas de evaluación BRISK
- `results/hog_vs_brisk_comparison.png` - Comparación visual de ambos modelos
- `results/test_*.png` - Imágenes de prueba con predicciones

## Conclusiones

### ✅ Aspectos Positivos
1. **Pipeline Completo Funcional**: Todo el flujo desde descarga de datos hasta inferencia funciona correctamente
2. **Automatización**: Script de descarga automática desde Kaggle
3. **Modularidad**: Código bien estructurado con separación clara de responsabilidades
4. **Configuración Flexible**: Uso de archivos YAML para configuración
5. **Logging Detallado**: Información completa durante todo el proceso
6. **Dos Modelos Implementados**: Características HOG y BRISK disponibles

### 📊 Rendimiento de los Modelos

#### Comparación HOG vs BRISK
| Métrica | HOG | BRISK | Ganador |
|---------|-----|-------|---------|
| IoU Promedio | 0.2998 | 0.2457 | **HOG** |
| IoU Mediana | 0.2561 | 0.1080 | **HOG** |
| IoU Máximo | 0.7818 | 0.9011 | **BRISK** |
| MAE | 0.0555 | 0.0667 | **HOG** |
| Precisión (IoU≥0.5) | 31.03% | 20.69% | **HOG** |
| Precisión (IoU≥0.75) | 4.60% | 6.90% | **BRISK** |
| Precisión (IoU≥0.9) | 0.00% | 1.15% | **BRISK** |
| Tamaño del modelo | 84.37 MB | 64.68 MB | **BRISK** |

**🏆 Ganador General: HOG** (4/7 métricas principales)

#### Observaciones
- **HOG**: Mejor rendimiento promedio y consistencia (IoU promedio más alto)
- **BRISK**: Mejor en casos extremos (IoU máximo más alto), modelo más ligero
- Ambos modelos muestran resultados moderados, lo cual es esperado dado que:
  - Es un enfoque basado en características tradicionales (no deep learning end-to-end)
  - Red neuronal totalmente conectada (no CNN)
  - Dataset relativamente pequeño (433 imágenes)

### 🔧 Mejoras Potenciales
1. Aumentar el dataset con data augmentation
2. Experimentar con diferentes arquitecturas de red
3. Probar con características BRISK además de HOG
4. Ajustar hiperparámetros (learning rate, arquitectura)
5. Implementar ensemble de modelos

## Instrucciones de Uso

### 1. Activar entorno virtual
```bash
source venv/bin/activate
```

### 2. Descargar datos (solo primera vez)
```bash
python scripts/download_data.py
```

### 3. Preparar dataset
```bash
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog \
    --output data/processed/detection_hog.pkl
```

### 4. Entrenar modelo
```bash
python scripts/train.py \
    --config configs/hog_config.yaml \
    --feature-type hog
```

### 5. Realizar inferencia
```bash
python scripts/inference.py \
    --model models/detection_hog_best.h5 \
    --image data/raw/images/Cars0.png \
    --feature-type hog \
    --output results/result.png
```

## Sistema de Prueba
- **OS**: Linux
- **Python**: 3.12
- **TensorFlow**: 2.13+
- **GPU**: No disponible (ejecutado en CPU)
- **Memoria**: Warnings sobre asignación >10% de memoria libre del sistema

## Estado del Proyecto

✅ **PROYECTO COMPLETAMENTE FUNCIONAL**

Todos los componentes han sido probados exitosamente:
- ✅ Descarga de datos desde Kaggle
- ✅ Preparación del dataset con extracción de características HOG
- ✅ Preparación del dataset con extracción de características BRISK
- ✅ Entrenamiento del modelo HOG con callbacks y métricas
- ✅ Entrenamiento del modelo BRISK con callbacks y métricas
- ✅ Evaluación de ambos modelos en conjunto de validación
- ✅ Inferencia en imágenes individuales
- ✅ Comparación de rendimiento HOG vs BRISK
- ✅ Visualización de resultados
