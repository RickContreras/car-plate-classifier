# Car Plate Classifier / Clasificador de Matrículas

Proyecto profesional para detección de placas vehiculares usando Redes Neuronales Fully Connected con características HOG y BRISK.

## 📋 Descripción

Este proyecto implementa un sistema de detección de bounding boxes usando:
- **Características HOG** (Histogram of Oriented Gradients)
- **Características BRISK** (Binary Robust Invariant Scalable Keypoints)
- **Redes Neuronales Fully Connected** para regresión de coordenadas

## 🏗️ Estructura del Proyecto

```
fc-detection-project/
├── src/
│   ├── features/           # Extracción de características
│   │   ├── __init__.py
│   │   ├── base.py        # Interfaz base
│   │   ├── hog.py         # Extractor HOG
│   │   └── brisk.py       # Extractor BRISK
│   ├── models/            # Arquitecturas de redes neuronales
│   │   ├── __init__.py
│   │   ├── fc_network.py  # Redes Fully Connected
│   │   └── layers.py      # Capas personalizadas
│   ├── data/              # Pipeline de datos
│   │   ├── __init__.py
│   │   ├── dataset.py     # Dataset loaders
│   │   ├── transforms.py  # Augmentaciones
│   │   └── utils.py       # Utilidades
│   ├── training/          # Sistema de entrenamiento
│   │   ├── __init__.py
│   │   ├── trainer.py     # Training loops
│   │   ├── callbacks.py   # Callbacks personalizados
│   │   └── losses.py      # Funciones de pérdida
│   └── evaluation/        # Métricas y evaluación
│       ├── __init__.py
│       ├── metrics.py     # IoU, MAE, etc.
│       └── visualize.py   # Visualización de resultados
├── configs/               # Archivos de configuración
│   ├── hog_config.yaml
│   └── brisk_config.yaml
├── scripts/               # Scripts de utilidad
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/                 # Tests unitarios
│   ├── test_features.py
│   ├── test_models.py
│   └── test_data.py
├── notebooks/             # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── docs/                  # Documentación
│   └── api.md
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/RickContreras/car-plate-classifier.git
cd car-plate-classifier
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar el paquete en modo desarrollo

```bash
pip install -e .
```

## 📊 Preparación de Datos

### Formato de Dataset

El proyecto espera imágenes con anotaciones en formato Pascal VOC XML:

```xml
<annotation>
  <filename>image.jpg</filename>
  <object>
    <name>licence</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>300</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```

### Preparar Dataset

```bash
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --output data/processed \
    --split 0.8
```

## 🎯 Entrenamiento

### Entrenar modelo HOG

```bash
python scripts/train.py --config configs/hog_config.yaml
```

### Entrenar modelo BRISK

```bash
python scripts/train.py --config configs/brisk_config.yaml
```

### Entrenar ambos modelos

```bash
python scripts/train.py --config configs/hog_config.yaml
python scripts/train.py --config configs/brisk_config.yaml
```

### Parámetros personalizados

```bash
python scripts/train.py \
    --feature-type hog \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 15
```

## 📈 Evaluación

```bash
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --feature-type hog \
    --data data/processed/test
```

## 🔮 Inferencia

```bash
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --feature-type hog \
    --image path/to/image.jpg \
    --output results/
```

## 📊 Métricas de Rendimiento

| Modelo | MAE | IoU Promedio | IoU > 0.5 | Parámetros |
|--------|-----|--------------|-----------|------------|
| HOG    | 7.45% | 39.55% | 48.3% | 4.3M |
| BRISK  | 6.89% | 17.20% | 10.3% | 439K |

## 🔧 Configuración

### Archivo de configuración (YAML)

```yaml
# configs/hog_config.yaml
feature_extractor:
  type: hog
  params:
    orientations: 9
    pixels_per_cell: 8
    cells_per_block: 3

model:
  architecture:
    - units: 512
      activation: relu
      batch_norm: true
      dropout: 0.3
    - units: 256
      activation: relu
      batch_norm: true
      dropout: 0.3
    - units: 128
      activation: relu
      batch_norm: true
      dropout: 0.2
    - units: 64
      activation: relu
      dropout: 0.2
    - units: 4
      activation: sigmoid

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  loss: mse
  callbacks:
    - type: early_stopping
      patience: 15
      monitor: val_loss
    - type: reduce_lr
      factor: 0.5
      patience: 7
    - type: model_checkpoint
      save_best_only: true
      monitor: val_avg_iou
```

## 🧪 Tests

Ejecutar todos los tests:

```bash
pytest tests/ -v
```

Ejecutar tests específicos:

```bash
pytest tests/test_features.py -v
pytest tests/test_models.py -v
```

Con cobertura:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 API Reference

### Feature Extractors

```python
from src.features import HOGFeatureExtractor, BRISKFeatureExtractor

# HOG
hog = HOGFeatureExtractor(orientations=9, pixels_per_cell=8)
features = hog.extract(image)

# BRISK
brisk = BRISKFeatureExtractor(n_keypoints=512)
features = brisk.extract(image)
```

### Models

```python
from src.models import FCNetwork

model = FCNetwork(
    input_dim=8100,
    architecture=[512, 256, 128, 64, 4],
    activations=['relu', 'relu', 'relu', 'relu', 'sigmoid']
)
model.compile(optimizer='adam', loss='mse')
```

### Training

```python
from src.training import Trainer

trainer = Trainer(model, config)
history = trainer.train(train_data, val_data)
```

## 🎨 Visualización

```python
from src.evaluation import visualize_predictions

visualize_predictions(
    model=model,
    images=test_images,
    ground_truth=test_boxes,
    save_path='results/predictions.png'
)
```

## 🤝 Contribuciones

1. Fork del proyecto
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📖 Referencias

- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection.
- Leutenegger, S., Chli, M., & Siegwart, R. Y. (2011). BRISK: Binary robust invariant scalable keypoints.
