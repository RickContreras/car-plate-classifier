# Documentación de la API

## Extractores de Características

### HOGFeatureExtractor

Extrae características de Histograma de Gradientes Orientados (HOG).

```python
from src.features import HOGFeatureExtractor

extractor = HOGFeatureExtractor(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    target_size=(200, 200)
)

features = extractor.extract(image)
```

**Parámetros:**
- `orientations` (int): Número de bins de orientación (por defecto: 9)
- `pixels_per_cell` (tuple): Tamaño de una celda en píxeles (por defecto: (8, 8))
- `cells_per_block` (tuple): Número de celdas en cada bloque (por defecto: (3, 3))
- `block_norm` (str): Método de normalización (por defecto: 'L2-Hys')
- `target_size` (tuple): Tamaño objetivo para redimensionar imagen (por defecto: (200, 200))
- `transform_sqrt` (bool): Aplicar compresión de ley de potencia (por defecto: True)

**Métodos:**
- `extract(image)`: Extraer características de una sola imagen
- `extract_batch(images)`: Extraer características de múltiples imágenes
- `get_feature_dim()`: Obtener dimensión del vector de características
- `visualize_hog(image)`: Visualizar características HOG

### BRISKFeatureExtractor

Extrae características BRISK (Binary Robust Invariant Scalable Keypoints).

```python
from src.features import BRISKFeatureExtractor

extractor = BRISKFeatureExtractor(
    n_keypoints=512,
    target_size=(200, 200),
    threshold=30
)

features = extractor.extract(image)
```

**Parámetros:**
- `n_keypoints` (int): Número de keypoints a extraer (por defecto: 512)
- `target_size` (tuple): Tamaño objetivo para redimensionar imagen (por defecto: (200, 200))
- `threshold` (int): Umbral de detección AGAST (por defecto: 30)
- `octaves` (int): Octavas de detección (por defecto: 3)
- `pattern_scale` (float): Escala de muestreo de patrón (por defecto: 1.0)

**Métodos:**
- `extract(image)`: Extraer características de una sola imagen
- `extract_batch(images)`: Extraer características de múltiples imágenes
- `get_feature_dim()`: Obtener dimensión del vector de características
- `visualize_keypoints(image)`: Visualizar keypoints detectados

## Modelos

### FCNetwork

Red Neuronal Completamente Conectada para regresión de bbox.

```python
from src.models import FCNetwork

model = FCNetwork(
    input_dim=8100,
    architecture=[512, 256, 128, 64, 4],
    use_batch_norm=True
)

model.compile(learning_rate=0.001)
```

**Parámetros:**
- `input_dim` (int): Dimensión de características de entrada
- `architecture` (list): Lista de tamaños de capas
- `activations` (list): Activación por capa (por defecto: relu + sigmoid)
- `use_batch_norm` (bool): Usar normalización por lotes (por defecto: True)
- `dropout_rates` (list): Tasa de dropout por capa
- `l2_reg` (float): Factor de regularización L2 (por defecto: 0.0)

**Métodos:**
- `compile(optimizer, learning_rate, loss, metrics)`: Compilar modelo
- `get_model()`: Obtener modelo de Keras
- `save(filepath)`: Guardar modelo
- `load(filepath)`: Cargar modelo (método estático)

## Datos

### DetectionDataset

Clase de dataset para tareas de detección.

```python
from src.data import DetectionDataset

dataset = DetectionDataset(
    features=features_array,
    bboxes=bboxes_array,
    image_paths=paths_list
)

train_ds, val_ds = dataset.split(train_ratio=0.8)
```

**Métodos:**
- `__len__()`: Obtener tamaño del dataset
- `__getitem__(idx)`: Obtener una muestra individual
- `split(train_ratio, shuffle, seed)`: Dividir en entrenamiento/validación
- `save(filepath)`: Guardar a archivo pickle
- `load(filepath)`: Cargar desde archivo pickle (método estático)

### Utilidades

```python
from src.data import normalize_bbox, denormalize_bbox

# Normalizar: píxeles -> [0, 1]
norm_bbox = normalize_bbox((xmin, ymin, xmax, ymax), img_w, img_h)

# Desnormalizar: [0, 1] -> píxeles
bbox = denormalize_bbox((x_c, y_c, w, h), img_w, img_h)
```

## Entrenamiento

### Trainer

Gestor de entrenamiento para modelos.

```python
from src.training import Trainer

trainer = Trainer(
    model=keras_model,
    save_dir='models',
    name='detection_hog'
)

history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

**Métodos:**
- `train(X_train, y_train, X_val, y_val, ...)`: Entrenar modelo
- `evaluate(X_test, y_test)`: Evaluar modelo
- `predict(X)`: Realizar predicciones
- `save_model(filepath)`: Guardar modelo entrenado
- `save_history(filepath)`: Guardar historial de entrenamiento

### Callbacks

```python
from src.training import get_callbacks

callbacks = get_callbacks(
    model_name='detection_hog',
    patience=15,
    reduce_lr_patience=7
)
```

Retorna lista con:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- TensorBoard

## Evaluación

### MetricsCalculator

Calcular métricas de detección.

```python
from src.evaluation import MetricsCalculator

calc = MetricsCalculator(iou_thresholds=[0.5, 0.75])
calc.update(y_true, y_pred)
metrics = calc.compute()
calc.print_metrics()
```

**Métricas:**
- `avg_iou`: IoU promedio
- `median_iou`: IoU mediana
- `mae`: Error Absoluto Medio
- `accuracy@threshold`: Precisión en umbral de IoU

### Funciones

```python
from src.evaluation import calculate_iou, evaluate_detections

# IoU individual
iou = calculate_iou(pred_bbox, true_bbox)

# Evaluación completa
metrics = evaluate_detections(y_true, y_pred, iou_threshold=0.5)
```

## Uso de Scripts

### Preparar Dataset

```bash
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog \
    --output data/processed/detection_hog.pkl
```

### Entrenar Modelo

```bash
python scripts/train.py \
    --config configs/hog_config.yaml \
    --dataset data/processed/detection_hog.pkl
```

### Evaluar Modelo

```bash
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --dataset data/processed/detection_hog.pkl \
    --feature-type hog
```

### Inferencia

```bash
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --image test_image.jpg \
    --feature-type hog \
    --output result.jpg \
    --show
```
