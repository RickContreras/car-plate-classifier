# 🧪 Pruebas Adicionales Recomendadas

## ✅ Pruebas Ya Completadas

### 1. Pruebas Unitarias
```bash
pytest tests/ -v --cov=src
```
- **Resultado**: 17/17 tests pasados (100%)
- **Cobertura**: 53%

### 2. Entrenamiento de Modelos
```bash
# HOG
python scripts/train.py --config configs/hog_config.yaml

# BRISK
python scripts/train.py --config configs/brisk_config.yaml
```
- **HOG**: 30% IoU, 31% precisión @0.5
- **BRISK**: 25% IoU, 21% precisión @0.5

### 3. Comparación de Modelos
```bash
python compare_models.py
```
- **Resultado**: HOG gana en 4/7 métricas
- **Archivos generados**: `model_comparison.png`

### 4. Benchmark de Velocidad
```bash
python benchmark_speed.py
```
- **HOG**: 111ms/imagen (~9 FPS)
- **BRISK**: 92ms/imagen (~11 FPS)
- **BRISK es 1.21x más rápido**

### 5. Análisis de Errores
```bash
python analyze_errors.py
```
- **Genera**: 
  - `analysis_best_hog.png` - Mejores predicciones HOG
  - `analysis_worst_hog.png` - Peores predicciones HOG
  - `analysis_best_brisk.png` - Mejores predicciones BRISK
  - `analysis_worst_brisk.png` - Peores predicciones BRISK

---

## 🔴 PRIORIDAD ALTA - Hacer Ahora

### 1. Prueba con Imágenes Nuevas (15 min)
**Objetivo**: Validar modelos con datos completamente nuevos

```bash
# Descargar imágenes de test desde internet
mkdir -p data/test_external
# Coloca 5-10 imágenes de autos con placas visibles

# Probar HOG
python scripts/inference.py \
    --model models/detection_hog_best.h5 \
    --image data/test_external/car1.jpg \
    --feature-type hog

# Probar BRISK
python scripts/inference.py \
    --model models/detection_brisk_best.h5 \
    --image data/test_external/car1.jpg \
    --feature-type brisk
```

**Qué buscar**:
- ¿Detecta la placa correctamente?
- ¿Cuál modelo funciona mejor?
- ¿Hay casos donde ambos fallan?

---

### 2. Test de Robustez (20 min)
**Objetivo**: Ver cómo responden los modelos a condiciones adversas

```bash
python test_robustness.py
```

**Genera visualizaciones** de cómo cada modelo responde a:
-   Imágenes oscuras/brillantes
-   Imágenes con ruido
-   Imágenes rotadas
-   Imágenes con blur
-   Diferentes escalas

**Archivos generados**:
- `robustness_test_hog.png`
- `robustness_test_brisk.png`

---

### 3. Análisis de Velocidad Detallado (10 min)
**Objetivo**: Identificar cuellos de botella

Crea `profile_speed.py`:
```python
import time
import cv2
from src.features.hog import HOGFeatureExtractor
from src.features.brisk import BRISKFeatureExtractor

img = cv2.imread('data/raw/Cars0.png')

# HOG
hog = HOGFeatureExtractor()
times = []
for _ in range(100):
    start = time.time()
    features = hog.extract(img)
    times.append(time.time() - start)
print(f"HOG: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")

# BRISK
brisk = BRISKFeatureExtractor()
times = []
for _ in range(100):
    start = time.time()
    features = brisk.extract(img)
    times.append(time.time() - start)
print(f"BRISK: {np.mean(times)*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
```

---

### 4. Validación Cruzada (30 min)
**Objetivo**: Asegurar que los modelos generalizan bien

Crea `cross_validation.py`:
```python
from sklearn.model_selection import KFold
import pickle
import numpy as np

# Cargar datos
with open('data/processed/detection_hog.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['features']
y = data['bboxes']

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ious = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Entrenar modelo
    # ... (usa el código de train.py)
    
    # Evaluar
    # ... (calcula IoU promedio)
    
print(f"\nIoU promedio: {np.mean(ious):.3f} ± {np.std(ious):.3f}")
```

---

## 🟡 PRIORIDAD MEDIA - Siguiente Fase

### 5. Data Augmentation (1-2 días)
**Impacto esperado**: +10-15% en métricas

Crea `augment_dataset.py`:
```python
import cv2
import numpy as np
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast,
    Rotate, GaussianBlur, ShiftScaleRotate
)

# Definir augmentations
transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    Rotate(limit=15, p=0.5),
    GaussianBlur(p=0.3),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
], bbox_params={'format': 'pascal_voc', 'label_fields': []})

# Aplicar a dataset
# Multiplicar dataset por 3-5x
```

Luego re-entrenar:
```bash
python scripts/prepare_dataset.py --augment
python scripts/train.py --config configs/hog_config.yaml --augmented
```

---

### 6. Ensemble de Modelos (4 horas)
**Impacto esperado**: +5-8% en métricas

Crea `ensemble.py`:
```python
# Cargar ambos modelos
hog_model = load_model('models/detection_hog_best.h5', compile=False)
brisk_model = load_model('models/detection_brisk_best.h5', compile=False)

# Predicción combinada
def ensemble_predict(image):
    hog_features = extract_hog(image)
    brisk_features = extract_brisk(image)
    
    hog_pred = hog_model.predict(hog_features)
    brisk_pred = brisk_model.predict(brisk_features)
    
    # Promedio ponderado (HOG más peso por ser más preciso)
    final_pred = 0.6 * hog_pred + 0.4 * brisk_pred
    return final_pred
```

---

### 7. Optimización de Hiperparámetros (1 día)
**Impacto esperado**: +2-10% en métricas

Crea `grid_search.py`:
```python
from sklearn.model_selection import ParameterGrid

# Definir grid
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [16, 32, 64],
    'dropout': [0.3, 0.4, 0.5],
    'hidden_units': [512, 1024, 2048]
}

best_iou = 0
best_params = None

for params in ParameterGrid(param_grid):
    print(f"\nProbando: {params}")
    
    # Entrenar con estos parámetros
    model = build_model(**params)
    history = train_model(model, X_train, y_train)
    
    # Evaluar
    iou = evaluate(model, X_val, y_val)
    
    if iou > best_iou:
        best_iou = iou
        best_params = params

print(f"\nMejores parámetros: {best_params}")
print(f"IoU: {best_iou:.3f}")
```

---

### 8. Transfer Learning (2-3 días)
**Impacto esperado**: +10-20% en métricas

```python
from tensorflow.keras.applications import ResNet50, MobileNetV2

# Usar features pre-entrenadas
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_resnet_features(image):
    # Preprocesar
    img = cv2.resize(image, (224, 224))
    img = preprocess_input(img)
    
    # Extraer features
    features = base_model.predict(np.expand_dims(img, 0))
    return features.flatten()
```

Luego entrenar FC network con estas features.

---

## 🟢 PRIORIDAD BAJA - Mejoras Futuras

### 9. Exportar para Producción
```bash
# TensorFlow Lite (móviles)
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/detection_hog_best.h5', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('models/detection_hog.tflite', 'wb').write(tflite_model)
"

# ONNX (interoperabilidad)
pip install tf2onnx
python -m tf2onnx.convert \
    --saved-model models/detection_hog_best.h5 \
    --output models/detection_hog.onnx
```

---

### 10. Crear API REST
```python
# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
model = load_model('models/detection_hog_best.h5', compile=False)

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
    
    features = extract_hog(img)
    bbox = model.predict(features)
    
    return jsonify({
        'bbox': bbox.tolist(),
        'confidence': float(calculate_confidence(bbox))
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

```bash
python app.py
# Test: curl -F "image=@test.jpg" http://localhost:5000/detect
```

---

### 11. Pipeline de Video
```python
# video_detection.py
import cv2

cap = cv2.VideoCapture('video.mp4')
model = load_model('models/detection_brisk_best.h5', compile=False)  # BRISK es más rápido

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detectar
    features = extract_brisk(frame)
    bbox = model.predict(features)
    
    # Dibujar
    draw_bbox(frame, bbox)
    cv2.imshow('Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Roadmap Sugerido (4 Semanas)

### Semana 1: Diagnóstico y Validación
- [ ] Análisis de errores ✅ (Completado)
- [ ] Test con imágenes nuevas
- [ ] Test de robustez
- [ ] Validación cruzada
- [ ] **Meta**: Entender limitaciones actuales

### Semana 2: Mejora de Datos
- [ ] Implementar data augmentation
- [ ] Re-entrenar ambos modelos
- [ ] Comparar resultados
- [ ] **Meta**: +10-15% en IoU

### Semana 3: Optimización
- [ ] Ensemble HOG + BRISK
- [ ] Grid search hiperparámetros
- [ ] Probar transfer learning
- [ ] **Meta**: +5-10% adicional en IoU

### Semana 4: Producción (Opcional)
- [ ] Exportar a TFLite/ONNX
- [ ] Crear API REST
- [ ] Pruebas de integración
- [ ] **Meta**: Sistema deployable

---

## 🎯 Objetivos de Mejora

| Métrica | Actual | Objetivo | Estrategia |
|---------|--------|----------|------------|
| IoU Promedio | 30% | 45-55% | Data augmentation + Transfer learning |
| Precisión @0.5 | 31% | 50-70% | Ensemble + Optimización |
| Velocidad | 9 FPS | 15-30 FPS | Cuantización + Optimización |
| Tamaño Modelo | 84 MB | <50 MB | Pruning + Cuantización |

---

## 💡 Tips Importantes

1. **Siempre guarda tus experimentos**:
   ```bash
   mkdir experiments
   cp models/detection_hog_best.h5 experiments/baseline_hog.h5
   ```

2. **Documenta tus resultados**:
   ```bash
   echo "Experimento: Data Augmentation 3x" >> experiments/log.txt
   echo "IoU: 0.42" >> experiments/log.txt
   ```

3. **Usa versionado**:
   ```bash
   git add .
   git commit -m "feat: Add data augmentation, IoU improved to 0.42"
   ```

4. **Compara siempre con baseline**:
   - No borres modelos anteriores
   - Mantén scripts de evaluación consistentes

---

## 🚀 Comando Rápido para Empezar

```bash
# Test rápido de todo
cd /home/rickcontreras/proyectos/car-plate-classifier/fc-detection-project
source venv/bin/activate

# 1. Análisis de errores (ya hecho)
python analyze_errors.py

# 2. Descargar una imagen de prueba
wget -O data/test_car.jpg "https://example.com/car_with_plate.jpg"

# 3. Probar inferencia
python scripts/inference.py \
    --model models/detection_hog_best.h5 \
    --image data/test_car.jpg \
    --feature-type hog

# 4. Ver resultados
ls -lh *.png
```

---

## ❓ FAQ

**P: ¿Por dónde empiezo?**  
R: Empieza con las pruebas de PRIORIDAD ALTA. Son rápidas (15-30 min) y te darán insights inmediatos.

**P: ¿Cuánto tiempo toma cada prueba?**  
- Análisis de errores: 5 min ✅
- Test con imágenes nuevas: 15 min
- Test de robustez: 20 min
- Validación cruzada: 30 min
- Data augmentation: 1-2 días
- Ensemble: 4 horas
- Transfer learning: 2-3 días

**P: ¿Qué prueba da más mejora?**  
R: Por experiencia: Transfer Learning > Data Augmentation > Ensemble > Grid Search

**P: ¿Necesito GPU?**  
R: No es obligatorio, pero ayuda mucho para transfer learning (10x más rápido).

---

¡Éxito con tus pruebas! 🚀
