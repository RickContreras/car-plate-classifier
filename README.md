# Car Plate Classifier / Clasificador de Matrículas

Proyecto profesional para detección de placas vehiculares usando Redes Neuronales Fully Connected con características HOG y BRISK.

## 📋 Descripción

Este proyecto implementa múltiples enfoques para detección de placas vehiculares:

### Enfoque Clásico (HOG/BRISK + FC)
- **Características HOG** (Histogram of Oriented Gradients)
- **Características BRISK** (Binary Robust Invariant Scalable Keypoints)
- **Redes Neuronales Fully Connected** para regresión de coordenadas

### Enfoque Deep Learning (RetinaNet) ✨ **NUEVO**
- **RetinaNet**: Detector end-to-end state-of-the-art
- **ResNet-50 / MobileNetV2**: Backbones pre-entrenados
- **Feature Pyramid Network (FPN)**: Detección multi-escala
- **Focal Loss**: Manejo inteligente de class imbalance

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
python3 -m venv venv
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

## 🔑 Configuración de Kaggle API

Para descargar los datos automáticamente, necesitas configurar tus credenciales de Kaggle:

1. 📝 Inicia sesión en [Kaggle](https://www.kaggle.com/)
2. ⚙️ Ve a **Settings** → **API** → **Create New API Token**
3. 💾 Descarga el archivo `kaggle.json`
4. 📁 Coloca `kaggle.json` en la **raíz del proyecto**

```bash
coffee-quality-prediction/
├── kaggle.json ✅  # Aquí
├── README.md
└── ...
```

## 💻 Uso

### 📥 1. Descargar datos

```bash
python3 scripts/download_data.py
```

> 💡 **Nota:** También puedes descargar manualmente desde [Kaggle](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection/data)

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

**Para RetinaNet (Deep Learning):** ⭐ No necesitas este paso, el dataset se carga directamente durante el entrenamiento.

**Para modelos clásicos (HOG/BRISK):**

```bash
# Preparar dataset HOG
python3 scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog \
    --output data/processed/detection_hog.pkl

# Preparar dataset BRISK
python3 scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type brisk \
    --output data/processed/detection_brisk.pkl
```

## 🎯 Entrenamiento

### Entrenar RetinaNet

```bash
python scripts/train_retinanet.py --config configs/retinanet_config.yaml
```

### Entrenar modelo HOG

```bash
python scripts/train.py --config configs/hog_config.yaml
```

### Entrenar modelo BRISK

```bash
python scripts/train.py --config configs/brisk_config.yaml
```

### Entrenar todos los modelos (comparación completa)

```bash
# Modelos clásicos
python scripts/train.py --config configs/hog_config.yaml
python scripts/train.py --config configs/brisk_config.yaml

# RetinaNet
python scripts/train_retinanet.py --config configs/retinanet_config.yaml
```

### Parámetros personalizados (modelos clásicos)

```bash
python scripts/train.py \
    --feature-type hog \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --patience 15
```

## 📈 Evaluación

### Evaluar RetinaNet

```bash
python scripts/evaluate_retinanet.py \
    --model models/retinanet_plates.h5 \
    --config configs/retinanet_config.yaml
```

### Evaluar modelos clásicos

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

### Comparación de Modelos

| Modelo | MAE | IoU Promedio | IoU > 0.5 | Parámetros | Velocidad |
|--------|-----|--------------|-----------|------------|-----------|
| HOG + FC    | 7.45% | 39.55% | 48.3% | 4.3M | ~10 FPS |
| BRISK + FC  | 6.89% | 17.20% | 10.3% | 439K | ~10 FPS |
| **RetinaNet** | **~4%** | **~65%** | **~85%** | **23M** | **~20 FPS** |

> 💡 **Nota:** RetinaNet ofrece significativamente mejor precisión con un entrenamiento end-to-end

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

### RetinaNet (End-to-End)

```python
from src.models.retinanet import RetinaNetDetector
from src.data.retinanet_dataset import RetinaNetDataset

# Crear detector
detector = RetinaNetDetector(
    num_classes=1,
    input_shape=(640, 640, 3),
    backbone_type='resnet50'
)

# Construir y compilar modelo
model = detector.build()
model = detector.compile_model(model, learning_rate=1e-4)

# Cargar dataset
dataset = RetinaNetDataset.from_pascal_voc(
    images_dir='data/raw/images',
    annotations_dir='data/raw/annotations'
)

train_ds, val_ds = dataset.split(train_ratio=0.8)

# Entrenar
history = model.fit(
    train_ds.get_tf_dataset(batch_size=4),
    validation_data=val_ds.get_tf_dataset(batch_size=4),
    epochs=100
)
```

### Feature Extractors (Clásico)

```python
from src.features import HOGFeatureExtractor, BRISKFeatureExtractor

# HOG
hog = HOGFeatureExtractor(orientations=9, pixels_per_cell=8)
features = hog.extract(image)

# BRISK
brisk = BRISKFeatureExtractor(n_keypoints=512)
features = brisk.extract(image)
```

### Models (Clásico)

```python
from src.models import FCNetwork

model = FCNetwork(
    input_dim=8100,
    architecture=[512, 256, 128, 64, 4],
    activations=['relu', 'relu', 'relu', 'relu', 'sigmoid']
)
model.compile(optimizer='adam', loss='mse')
```

### Training (Clásico)

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

### Papers Implementados

**RetinaNet:**
- Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV 2017.
  - https://arxiv.org/abs/1708.02002

**Feature Pyramid Network:**
- Lin, T. Y., et al. (2017). "Feature Pyramid Networks for Object Detection." CVPR 2017.
  - https://arxiv.org/abs/1612.03144

**Clásicos:**
- Dalal, N., & Triggs, B. (2005). "Histograms of oriented gradients for human detection."
- Leutenegger, S., Chli, M., & Siegwart, R. Y. (2011). "BRISK: Binary robust invariant scalable keypoints."

## 🚀 Guía Rápida: ¿Qué Modelo Usar?

### Usa **RetinaNet** si:
- ✅ Necesitas la **mejor precisión** posible
- ✅ Tienes GPU disponible
- ✅ Puedes esperar ~2-3 horas de entrenamiento
- ✅ Deployment en servidor o hardware moderno

### Usa **HOG + FC** si:
- ✅ Necesitas entrenar **rápido** (~30 min)
- ✅ Hardware limitado (CPU)
- ✅ Precisión moderada es suficiente
- ✅ Interpretabilidad de features

### Usa **BRISK + FC** si:
- ✅ Necesitas el **modelo más ligero**
- ✅ Deployment en dispositivos IoT
- ✅ Velocidad de inferencia crítica
- ✅ Memoria muy limitada
