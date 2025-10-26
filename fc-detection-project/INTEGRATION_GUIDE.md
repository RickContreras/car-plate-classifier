# Guía de Integración con tus Datos Existentes

## 🎯 Objetivo

Este proyecto está diseñado para entrenar modelos de detección de placas vehiculares usando **Redes Fully Connected** con características **HOG** y **BRISK**.

## 📊 Tus Datos Actuales

Actualmente tienes:
- **433 imágenes** en `data/images/`
- **Anotaciones XML** (Pascal VOC) en `data/annotations/`
- Ya entrenaste YOLO, NN-HOG y NN-BRISK en el proyecto principal

## 🚀 Cómo Usar Este Proyecto

### Opción 1: Entrenar desde Cero (Recomendado para Aprendizaje)

```bash
# 1. Instalar el proyecto
cd fc-detection-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Copiar tus datos existentes
mkdir -p data/raw/images
mkdir -p data/raw/annotations

# Desde el directorio principal del proyecto car-plate-classifier:
cp ../data/images/* data/raw/images/
cp ../data/annotations/* data/raw/annotations/

# 3. Preparar datasets
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog

python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type brisk

# 4. Entrenar modelos
python scripts/train.py --config configs/hog_config.yaml
python scripts/train.py --config configs/brisk_config.yaml

# 5. Evaluar
python scripts/evaluate.py --model models/detection_hog.h5 --feature-type hog
python scripts/evaluate.py --model models/detection_brisk.h5 --feature-type brisk

# 6. Inferencia en una imagen
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --image data/raw/images/Cars0.png \
    --feature-type hog \
    --show
```

### Opción 2: Pipeline Completo Automatizado

```bash
# Ejecutar todo el pipeline de una vez
./run_pipeline.sh
```

### Opción 3: Usar el Código Existente

Si prefieres usar el código que ya tienes funcionando, puedes:

1. **Copiar módulos específicos**:
   ```bash
   # Copiar extractores de características al proyecto principal
   cp fc-detection-project/src/features/*.py ../src/
   ```

2. **Usar como referencia**: Este proyecto muestra las mejores prácticas para estructurar un proyecto ML profesional.

## 📁 Estructura de Datos Esperada

```
fc-detection-project/
└── data/
    └── raw/
        ├── images/
        │   ├── Cars0.png
        │   ├── Cars1.png
        │   └── ...
        └── annotations/
            ├── Cars0.xml
            ├── Cars1.xml
            └── ...
```

Formato XML (Pascal VOC):
```xml
<annotation>
  <filename>Cars0.png</filename>
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

## 🔧 Personalización

### Cambiar Arquitectura del Modelo

Edita `configs/hog_config.yaml`:

```yaml
model:
  architecture:
    - units: 1024    # Aumentar capacidad
      activation: relu
      batch_norm: true
      dropout: 0.4
    - units: 512
      activation: relu
      batch_norm: true
      dropout: 0.3
    # ... más capas
```

### Ajustar Hiperparámetros de Entrenamiento

```yaml
training:
  epochs: 150
  batch_size: 64
  learning_rate: 0.0005
  
  callbacks:
    early_stopping:
      patience: 20
```

### Cambiar Parámetros de HOG

```yaml
feature_extractor:
  params:
    orientations: 12        # Más bins de orientación
    pixels_per_cell: [16, 16]  # Células más grandes
    target_size: [256, 256]    # Mayor resolución
```

## 📊 Resultados Esperados

Basado en tu entrenamiento anterior:

| Modelo | MAE | IoU Promedio | IoU > 0.5 |
|--------|-----|--------------|-----------|
| HOG    | ~7.5% | ~40% | ~48% |
| BRISK  | ~6.9% | ~17% | ~10% |

**HOG** generalmente supera a **BRISK** para este tipo de tarea.

## 🎓 Diferencias con tu Código Actual

### Ventajas de Este Proyecto:

1. **Modularidad**: Código organizado en módulos reutilizables
2. **Configuración**: Todo configurable vía YAML
3. **Tests**: Suite completa de tests unitarios
4. **Documentación**: API docs, quickstart, ejemplos
5. **Reproducibilidad**: Configuraciones guardadas con modelos
6. **Escalabilidad**: Fácil agregar nuevos extractores o modelos
7. **Profesionalismo**: Sigue mejores prácticas de la industria

### Tu Código Actual:

- ✅ Funcional y efectivo
- ✅ Resultados demostrados
- ⚠️ Menos estructurado para compartir/mantener

## 🔄 Migrar Modelos Existentes

Si ya tienes modelos entrenados que quieres usar aquí:

```python
# Copiar modelos
cp ../models/detection_nn/detection_nn_hog.h5 fc-detection-project/models/
cp ../models/detection_nn/detection_nn_brisk.h5 fc-detection-project/models/

# Usar con scripts de este proyecto
python scripts/evaluate.py \
    --model models/detection_nn_hog.h5 \
    --dataset data/processed/detection_hog.pkl \
    --feature-type hog
```

## 📚 Aprender y Experimentar

### Para Entender Mejor:

1. **Explora los ejemplos**:
   ```bash
   python examples/usage_example.py
   ```

2. **Lee la documentación**:
   - `README.md`: Visión general
   - `docs/QUICKSTART.md`: Inicio rápido
   - `docs/api.md`: Referencia API
   - `STRUCTURE.md`: Estructura del proyecto

3. **Ejecuta los tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Experimenta en notebooks**:
   ```bash
   jupyter notebook notebooks/
   ```

## 🎯 Casos de Uso

### 1. Proyecto de Portfolio
- Demuestra código limpio y bien documentado
- Muestra conocimiento de mejores prácticas
- Fácil de compartir en GitHub

### 2. Base para Otros Proyectos
- Reutiliza extractores de características
- Adapta para otras tareas de detección
- Extensible a nuevos datasets

### 3. Comparación de Métodos
- Benchmark HOG vs BRISK
- Diferentes arquitecturas
- Análisis de rendimiento

### 4. Producción
- Código listo para deployment
- Tests aseguran calidad
- Configuraciones reproducibles

## ⚡ Quick Start (5 minutos)

```bash
# 1. Setup
cd fc-detection-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Ver ejemplos de uso
python examples/usage_example.py

# 3. Si tienes datos, entrenar HOG (más rápido que BRISK)
# Copiar datos primero, luego:
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog

python scripts/train.py --config configs/hog_config.yaml

# 4. Inferencia
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --image data/raw/images/Cars0.png \
    --feature-type hog \
    --show
```

## 🤝 Contribuir

Este es un proyecto modular y extensible. Puedes:

- Agregar nuevos extractores de características
- Implementar diferentes arquitecturas de red
- Mejorar callbacks y métricas
- Agregar más tests
- Mejorar documentación

## 📞 Soporte

Si tienes dudas sobre cómo usar este proyecto con tus datos:

1. Lee `docs/QUICKSTART.md`
2. Revisa `examples/usage_example.py`
3. Consulta `docs/api.md`
4. Ejecuta los tests para verificar instalación

## 🎉 Conclusión

Este proyecto te da:
- ✅ Código limpio y profesional
- ✅ Estructura escalable
- ✅ Documentación completa
- ✅ Tests verificados
- ✅ Configuración flexible
- ✅ Listo para portfolio o producción

¡Disfruta experimentando con diferentes configuraciones y mejorando los modelos! 🚀
