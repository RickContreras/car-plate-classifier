# 🚀 Guía de Inicio Rápido

## 📋 Requisitos Previos

- Python 3.12.3 o superior
- pip (gestor de paquetes de Python)
- 4GB de RAM mínimo
- 2GB de espacio en disco

## 🔧 Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/RickContreras/car-plate-classifier.git
```

## 🐍 Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows
```

## 📦 Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Paso 4: Descargar Dataset

**Opción A: Usando script automático (Requiere cuenta de Kaggle)**

```bash
# Configurar credenciales de Kaggle
# 1. Ir a https://www.kaggle.com/settings/account
# 2. Crear API Token (descarga kaggle.json)
# 3. Colocar en ~/.kaggle/kaggle.json

# Descargar dataset
python scripts/download_data.py
```

**Opción B: Descarga manual**

1. Ir a [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
2. Descargar y extraer en `data/raw/`
3. Verificar que existan:
   - `data/raw/images/` (con imágenes .png)
   - `data/raw/annotations/` (con archivos .xml)

## 🎯 Paso 5: Preparar Datasets

```bash
# Preparar features HOG
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog \
    --output data/processed/detection_hog.pkl

# Preparar features BRISK
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type brisk \
    --output data/processed/detection_brisk.pkl
```

**Tiempo estimado:** 3-5 minutos por dataset

## 🤖 Paso 6: Entrenar Modelos

```bash
# Entrenar modelo HOG
python scripts/train.py --config configs/hog_config.yaml

# Entrenar modelo BRISK
python scripts/train.py --config configs/brisk_config.yaml
```

**Tiempo estimado:** 10-30 minutos por modelo (dependiendo del hardware)

## 🎨 Paso 7: Ejecutar Aplicación Web (Gradio)

```bash
python app_gradio.py
```

Abrir navegador en: `http://127.0.0.1:7861`

## ✅ Verificación Rápida

```bash
# Ejecutar pruebas unitarias
pytest tests/ -v

# Verificar modelo HOG
python scripts/inference.py \
    --model models/detection_hog_best.h5 \
    --image data/raw/images/Cars0.png \
    --feature-type hog \
    --show

# Comparar modelos
python compare_models.py
```

---

## 📂 Estructura de Archivos Esperada

Después de completar todos los pasos:

```
fc-detection-project/
├── venv/                           # Entorno virtual
├── data/
│   ├── raw/
│   │   ├── images/                 # 433 imágenes .png
│   │   └── annotations/            # 433 archivos .xml
│   └── processed/
│       ├── detection_hog.pkl       # Features HOG (433 samples)
│       └── detection_brisk.pkl     # Features BRISK (433 samples)
├── models/
│   ├── detection_hog_best.h5       # Modelo HOG entrenado (~254 MB)
│   ├── detection_brisk_best.h5     # Modelo BRISK entrenado (~195 MB)
│   ├── hog_training_history.png    # Gráficas de entrenamiento
│   └── brisk_training_history.png
└── results/                        # Resultados de inferencia
```

---

## 🎯 Comandos Más Importantes (Resumen)

### Para empezar desde cero:
```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Datos
python scripts/download_data.py

# 3. Preparar
python scripts/prepare_dataset.py --images data/raw/images --annotations data/raw/annotations --feature-type hog
python scripts/prepare_dataset.py --images data/raw/images --annotations data/raw/annotations --feature-type brisk

# 4. Entrenar
python scripts/train.py --config configs/hog_config.yaml
python scripts/train.py --config configs/brisk_config.yaml

# 5. Usar
python app_gradio.py
```

### Si los modelos ya están entrenados:
```bash
# Solo ejecutar la aplicación
source venv/bin/activate
python app_gradio.py
```

---

## 🔍 Comandos Adicionales Útiles

### Evaluación
```bash
# Evaluar modelo HOG
python scripts/evaluate.py \
    --model models/detection_hog_best.h5 \
    --data data/processed/detection_hog.pkl \
    --feature-type hog

# Comparar ambos modelos
python compare_models.py
```

### Inferencia Individual
```bash
# Con HOG
python scripts/inference.py \
    --model models/detection_hog_best.h5 \
    --image TU_IMAGEN.jpg \
    --feature-type hog \
    --output resultado_hog.jpg \
    --show

# Con BRISK
python scripts/inference.py \
    --model models/detection_brisk_best.h5 \
    --image TU_IMAGEN.jpg \
    --feature-type brisk \
    --output resultado_brisk.jpg \
    --show
```

---

## 🐛 Solución de Problemas Comunes

### Error: "Out of memory"
Reduce el batch size en los archivos de configuración (`configs/*.yaml`):
```yaml
training:
  batch_size: 16  # Cambiar a 8 o 4 si hay problemas de memoria
```

### Los modelos predicen mal
Esto es normal - el dataset es pequeño (433 imágenes). 
Mejoras recomendadas:
1. Data augmentation
2. Transfer learning
3. Más datos de entrenamiento

---

## 📊 Resultados Esperados

### Métricas de los Modelos:

| Modelo | IoU Promedio | Precisión @0.5 | Velocidad | Tamaño |
|--------|--------------|----------------|-----------|--------|
| HOG    | ~30%         | ~31%           | 112 ms    | 254 MB |
| BRISK  | ~25%         | ~21%           | 93 ms     | 195 MB |

**Nota:** Estos son resultados con el dataset base. Se pueden mejorar significativamente con data augmentation y transfer learning.

---

## 💡 Siguientes Pasos

Después de completar la instalación básica:

1. **Probar la aplicación web** (`app_gradio.py`)
2. **Revisar documentación** en `docs/` y `PRUEBAS_RECOMENDADAS.md`

---

## 📞 Soporte

Para problemas o preguntas:
- Revisar `STRUCTURE.md` para entender la arquitectura
- Consultar `INTEGRATION_GUIDE.md` para integración

---

**¡Listo para detectar placas vehiculares! 🚗**
