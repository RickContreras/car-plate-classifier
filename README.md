# 🚗 Clasificador de Placas Vehiculares

Proyecto de visión por computadora para detección y clasificación de placas vehiculares usando descriptores **HOG** (Histogram of Oriented Gradients) y **BRISK** (Binary Robust Invariant Scalable Keypoints).

## 📋 Características

- ✅ Preprocesamiento automático de imágenes
- ✅ Extracción de características con HOG y BRISK
- ✅ Múltiples modelos de clasificación (SVM, Random Forest, Redes Neuronales)
- ✅ Interfaz gráfica intuitiva para predicción
- ✅ Métricas de evaluación completas (Accuracy, Precision, Recall, F1-Score)
- ✅ Matrices de confusión y visualizaciones
- ✅ Manejo de clases balanceadas (muestras positivas y negativas)

## 🛠️ Instalación

### Requisitos Previos
- Python 3.8 o superior
- pip
- virtualenv (opcional pero recomendado)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/RickContreras/car-plate-classifier.git
cd car-plate-classifier
```

2. **Crear y activar entorno virtual**
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar el dataset**
```bash
python scripts/download_data.py
```

## 📂 Estructura del Proyecto

```
car-plate-classifier/
├── app/
│   ├── __init__.py
│   └── gui.py                 # Interfaz gráfica
├── config/
│   └── config.yaml            # Configuración del proyecto
├── data/
│   ├── raw/                   # Datos originales
│   │   ├── images/
│   │   └── annotations/
│   └── processed/             # Datos procesados
├── models/                    # Modelos entrenados
├── notebooks/                 # Jupyter notebooks
├── results/                   # Resultados y visualizaciones
├── scripts/
│   ├── __init__.py
│   └── download_data.py       # Script de descarga de datos
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Preprocesamiento de imágenes
│   ├── feature_extraction.py # Extracción de características
│   └── train_models.py        # Entrenamiento de modelos
├── tests/
│   └── __init__.py
├── main.py                    # Script principal
├── requirements.txt
├── kaggle.json               # Credenciales de Kaggle
└── README.md
```

## 🚀 Uso

### 1. Entrenamiento de Modelos

Ejecuta el pipeline completo de entrenamiento:

```bash
python main.py
```

Este script ejecutará:
1. Preprocesamiento del dataset
2. Extracción de características (HOG y BRISK)
3. Entrenamiento de 6 modelos diferentes:
   - SVM con HOG
   - SVM con BRISK
   - Random Forest con HOG
   - Random Forest con BRISK
   - Red Neuronal con HOG
   - Red Neuronal con BRISK

### 2. Interfaz Gráfica

Una vez entrenados los modelos, lanza la interfaz gráfica:

```bash
python app/gui.py
```

**Funcionalidades de la GUI:**
- Cargar y visualizar imágenes
- Seleccionar modelo entrenado
- Clasificar imágenes en tiempo real
- Ver resultados con nivel de confianza

### 3. Uso Individual de Módulos

**Preprocesamiento:**
```bash
python src/preprocessing.py
```

**Extracción de características:**
```bash
python src/feature_extraction.py
```

**Entrenamiento:**
```bash
python src/train_models.py
```

## ⚙️ Configuración

El archivo `config/config.yaml` permite personalizar:

```yaml
data:
  img_size: [128, 128]        # Tamaño de redimensionamiento
  test_size: 0.2              # Proporción de datos de prueba
  
preprocessing:
  resize: true                # Redimensionar imágenes
  grayscale: true             # Convertir a escala de grises
  normalize: true             # Normalizar valores
  equalize_hist: false        # Ecualización de histograma

features:
  hog:
    orientations: 9
    pixels_per_cell: [8, 8]
    cells_per_block: [2, 2]
  
  brisk:
    threshold: 30
    octaves: 3
    pattern_scale: 1.0
```

## 📊 Resultados

Los modelos generan:
- **Matrices de confusión** guardadas en `results/`
- **Historial de entrenamiento** (para redes neuronales)
- **Métricas de evaluación** en consola

Ejemplo de métricas:
```
SVM_HOG:
   • accuracy: 0.9500
   • precision: 0.9400
   • recall: 0.9600
   • f1_score: 0.9500
```

## 🧪 Testing

Ejecuta los tests con:
```bash
pytest tests/
```

## 📝 Dataset

El proyecto utiliza el dataset de placas vehiculares disponible en Kaggle. Asegúrate de:
1. Tener una cuenta en Kaggle
2. Configurar `kaggle.json` con tus credenciales
3. Ejecutar el script de descarga

## 🤝 Contribuidores

- [Ricardo Contreras Garzón](https://github.com/RickContreras)
- [Maria Cristina Vergara](https://github.com/cristinavergara1)
- [Santiago Graciano](https://github.com/santiagogracianod)

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

## 🙏 Agradecimientos

- Dataset de placas vehiculares de Kaggle
- Librerías: OpenCV, scikit-learn, scikit-image, TensorFlow

---

**¿Necesitas ayuda?** Abre un issue en el repositorio.