# 🎉 Proyecto de Clasificación de Placas Vehiculares - Estado Actual

## ✅ Completado

### 1. Entorno Virtual
- ✅ Creado en `venv/`
- ✅ Python 3.12.3
- ✅ Todas las dependencias instaladas correctamente

### 2. Dependencias Instaladas
```
- opencv-python==4.10.0.84
- scikit-learn==1.5.2
- scikit-image==0.24.0
- numpy==1.26.4
- pandas==2.2.3
- matplotlib==3.9.2
- seaborn==0.13.2
- tensorflow==2.18.0
- Pillow==10.4.0
- PyYAML==6.0.2
- lxml==5.3.0
- kaggle==1.7.4.5
```

### 3. Dataset Descargado
- ✅ **433 imágenes** de vehículos (.png)
- ✅ **433 anotaciones** XML con bounding boxes de placas
- ✅ Total: 203.02 MB
- ✅ Ubicación: `data/raw/`

### 4. Módulos Implementados

#### `src/preprocessing.py`
- ✅ Carga de configuración desde YAML
- ✅ Parsing de anotaciones XML
- ✅ Recorte de regiones de placas
- ✅ Preprocesamiento de imágenes (resize, grayscale, normalización)
- ✅ Generación de muestras negativas para balanceo
- ✅ Split automático train/test

#### `src/feature_extraction.py`
- ✅ Clase `HOGFeatureExtractor` (Histogram of Oriented Gradients)
- ✅ Clase `BRISKFeatureExtractor` (Binary Robust Invariant Scalable Keypoints)
- ✅ Procesamiento en lote de imágenes
- ✅ Vectores de características de tamaño fijo

#### `src/train_models.py`
- ✅ Clase `PlateClassifier` con 3 tipos de modelos:
  - SVM (Support Vector Machine)
  - Random Forest
  - Redes Neuronales (TensorFlow/Keras)
- ✅ Evaluación automática con métricas completas
- ✅ Generación de matrices de confusión
- ✅ Visualización del historial de entrenamiento
- ✅ Guardado/carga de modelos

#### `app/gui.py`
- ✅ Interfaz gráfica completa con Tkinter
- ✅ Carga y visualización de imágenes
- ✅ Selección de modelos entrenados
- ✅ Clasificación en tiempo real
- ✅ Resultados con nivel de confianza

### 5. Scripts Disponibles

- ✅ `main.py` - Pipeline completo de entrenamiento
- ✅ `example_usage.py` - Uso sin GUI
- ✅ `test_modules.py` - Tests de validación
- ✅ `scripts/download_data.py` - Descarga de dataset

### 6. Estructura de Directorios
```
car-plate-classifier/
├── venv/                    ✅ Entorno virtual activo
├── data/
│   ├── raw/
│   │   ├── images/         ✅ 433 imágenes
│   │   └── annotations/    ✅ 433 archivos XML
│   └── processed/          ✅ Listo para uso
├── models/                 ✅ Listo para guardar modelos
├── results/                ✅ Listo para resultados
├── src/                    ✅ Todos los módulos implementados
├── app/                    ✅ GUI implementada
├── scripts/                ✅ Scripts de utilidad
└── config/config.yaml      ✅ Configuración lista
```

## 🚀 Próximos Pasos

### Opción 1: Entrenar Todos los Modelos (Recomendado)
```bash
source venv/bin/activate
python main.py
```
Esto entrenará 6 modelos:
- SVM + HOG
- SVM + BRISK
- Random Forest + HOG
- Random Forest + BRISK
- Red Neuronal + HOG
- Red Neuronal + BRISK

**Tiempo estimado**: 30-60 minutos (dependiendo del hardware)

### Opción 2: Usar la Interfaz Gráfica (requiere modelos entrenados)
```bash
source venv/bin/activate
python app/gui.py
```

### Opción 3: Clasificar desde Línea de Comandos
```bash
source venv/bin/activate
# Imagen individual
python example_usage.py --image ruta/imagen.jpg --model models/svm_hog.pkl

# Directorio completo
python example_usage.py --dir test_images/ --model models/neural_network_hog.h5 --model-type neural_network
```

## 📊 Características del Proyecto

### Preprocesamiento
- Redimensionado a 128x128 píxeles
- Conversión a escala de grises
- Normalización de valores
- Balanceo automático de clases (positivas/negativas)

### Extracción de Características
- **HOG**: 8100 características (9 orientaciones, 8x8 pixels/cell, 2x2 cells/block)
- **BRISK**: 512 características (threshold=30, octaves=3)

### Modelos de Clasificación
- **SVM**: Kernel RBF, C=1.0
- **Random Forest**: 100 estimadores, max_depth=20
- **Red Neuronal**: 4 capas densas (256→128→64→1) con Dropout

### Métricas de Evaluación
- Accuracy
- Precision
- Recall
- F1-Score
- Matrices de Confusión
- Historial de entrenamiento (Redes Neuronales)

## 📝 Notas Técnicas

### Rendimiento
- CPU: El proyecto funciona perfectamente en CPU
- GPU: TensorFlow detectó que no hay GPU disponible pero funcionará en CPU
- Split: 80% entrenamiento, 20% prueba

### Dataset
- 433 imágenes de vehículos con placas anotadas
- Formato: PNG
- Anotaciones: Pascal VOC XML
- Origen: Kaggle (andrewmvd/car-plate-detection)

## ✅ Estado de Pruebas
Todos los módulos fueron probados y funcionan correctamente:
- ✅ Carga de configuración
- ✅ Preprocesamiento de imágenes
- ✅ Extracción de características HOG
- ✅ Extracción de características BRISK
- ✅ Creación de modelos SVM
- ✅ Creación de modelos Random Forest
- ✅ Creación de Redes Neuronales
- ✅ Entrenamiento y predicción
- ✅ Parsing de XML reales del dataset

---

**Fecha de creación**: 26 de octubre de 2025
**Estado**: ✅ Listo para entrenamiento
