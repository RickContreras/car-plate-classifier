# Estructura del Proyecto FC Detection

## Visión General

```
fc-detection-project/
│
├── 📁 src/                          # Código fuente principal
│   ├── __init__.py
│   ├── 📁 features/                 # Extracción de características
│   │   ├── __init__.py
│   │   ├── base.py                 # Clase base abstracta
│   │   ├── hog.py                  # Extractor HOG
│   │   └── brisk.py                # Extractor BRISK
│   │
│   ├── 📁 models/                   # Arquitecturas de redes
│   │   ├── __init__.py
│   │   └── fc_network.py           # Red Fully Connected
│   │
│   ├── 📁 data/                     # Pipeline de datos
│   │   ├── __init__.py
│   │   ├── dataset.py              # Clase DetectionDataset
│   │   └── utils.py                # Utilidades (normalize_bbox, etc.)
│   │
│   ├── 📁 training/                 # Sistema de entrenamiento
│   │   ├── __init__.py
│   │   ├── trainer.py              # Clase Trainer
│   │   └── callbacks.py            # Callbacks personalizados
│   │
│   └── 📁 evaluation/               # Métricas y evaluación
│       ├── __init__.py
│       └── metrics.py              # IoU, MAE, etc.
│
├── 📁 configs/                      # Archivos de configuración
│   ├── hog_config.yaml             # Config para HOG
│   └── brisk_config.yaml           # Config para BRISK
│
├── 📁 scripts/                      # Scripts ejecutables
│   ├── prepare_dataset.py          # Preparar dataset
│   ├── train.py                    # Entrenar modelos
│   ├── evaluate.py                 # Evaluar modelos
│   └── inference.py                # Inferencia
│
├── 📁 tests/                        # Tests unitarios
│   ├── test_features.py            # Tests de extractores
│   ├── test_models.py              # Tests de modelos
│   └── test_data.py                # Tests de datos
│
├── 📁 docs/                         # Documentación
│   ├── api.md                      # Documentación API
│   └── QUICKSTART.md               # Guía rápida
│
├── 📁 examples/                     # Ejemplos de uso
│   └── usage_example.py            # Ejemplo completo
│
├── 📁 notebooks/                    # Jupyter notebooks
│   └── (para análisis exploratorio)
│
├── 📄 README.md                     # Documentación principal
├── 📄 requirements.txt              # Dependencias Python
├── 📄 setup.py                      # Instalación del paquete
├── 📄 setup.cfg                     # Configuración de herramientas
├── 📄 .gitignore                    # Archivos ignorados por Git
├── 📄 LICENSE                       # Licencia MIT
└── 📄 run_pipeline.sh               # Script para pipeline completo
```

## Componentes Principales

### 1. Feature Extraction (`src/features/`)
- **base.py**: Interfaz abstracta para extractores
- **hog.py**: Implementación HOG (8100 features)
- **brisk.py**: Implementación BRISK (32768 features)

### 2. Models (`src/models/`)
- **fc_network.py**: Red Fully Connected configurable
  - Arquitectura: [512, 256, 128, 64, 4]
  - Batch Normalization
  - Dropout regularization
  - L2 regularization opcional

### 3. Data (`src/data/`)
- **dataset.py**: Clase DetectionDataset
  - Almacena features + bboxes
  - Split train/val
  - Save/Load pickle
- **utils.py**: Utilidades
  - normalize_bbox(): pixels → [0,1]
  - denormalize_bbox(): [0,1] → pixels
  - parse_pascal_voc(): Lee XMLs

### 4. Training (`src/training/`)
- **trainer.py**: Clase Trainer
  - Training loop
  - Evaluation
  - Model saving
  - History logging
- **callbacks.py**: Callbacks
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint
  - TensorBoard
  - IoUCallback (custom)

### 5. Evaluation (`src/evaluation/`)
- **metrics.py**: Métricas
  - IoU calculation
  - MAE
  - Accuracy@threshold
  - MetricsCalculator class

## Flujo de Trabajo

```
1. Preparar Datos
   ├── Imágenes + XMLs (Pascal VOC)
   └── scripts/prepare_dataset.py
       └── features.pkl + bboxes.pkl

2. Entrenar Modelo
   ├── Cargar dataset.pkl
   ├── Crear FCNetwork
   ├── Configurar callbacks
   └── scripts/train.py
       └── model.h5 + history.json

3. Evaluar Modelo
   ├── Cargar model.h5
   ├── Cargar test data
   └── scripts/evaluate.py
       └── metrics.json

4. Inferencia
   ├── Cargar model.h5
   ├── Extraer features de imagen
   └── scripts/inference.py
       └── imagen con bbox dibujado
```

## Configuración YAML

Cada modelo tiene su config YAML con:
- Feature extractor params
- Model architecture
- Training hyperparameters
- Callbacks configuration
- Paths

## Tests

Cobertura completa con pytest:
- test_features.py: HOG y BRISK extractors
- test_models.py: FCNetwork
- test_data.py: Dataset y utils

## Documentación

- **README.md**: Visión general y ejemplos
- **docs/api.md**: Referencia completa de API
- **docs/QUICKSTART.md**: Guía de inicio rápido
- **examples/**: Código de ejemplo

## Mejores Prácticas Implementadas

✅ Modularidad: Cada componente es independiente
✅ Configurabilidad: YAML configs
✅ Testabilidad: Tests unitarios
✅ Documentación: Docstrings y docs/
✅ Reproducibilidad: Seeds, configs guardados
✅ Logging: TensorBoard, history JSON
✅ Type hints: Anotaciones de tipos
✅ Error handling: Validaciones y excepciones
✅ Code style: PEP 8, docstrings
✅ Versioning: Setup.py, __version__

## Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `setup.py` | Instalación del paquete |
| `requirements.txt` | Dependencias |
| `configs/*.yaml` | Configuración de modelos |
| `run_pipeline.sh` | Pipeline automatizado |
| `examples/usage_example.py` | Ejemplos de uso |
| `.gitignore` | Archivos a ignorar |
| `LICENSE` | Licencia MIT |
