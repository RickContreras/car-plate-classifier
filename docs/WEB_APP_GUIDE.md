# 🌐 Aplicación Web - Clasificador de Placas

## 🎯 Descripción

Aplicación web moderna construida con **Gradio** que permite clasificar imágenes para detectar si contienen placas vehiculares.

## ✨ Características

- ✅ **Interfaz web intuitiva** - Accesible desde cualquier navegador
- ✅ **6 modelos disponibles** - Elige entre SVM, Random Forest y Neural Networks
- ✅ **Múltiples fuentes de entrada** - Sube imagen, usa webcam o portapapeles
- ✅ **Resultados en tiempo real** - Clasificación instantánea con nivel de confianza
- ✅ **Visualización de métricas** - Ve el rendimiento de cada modelo
- ✅ **Responsive** - Funciona en desktop, tablet y móvil

## 🚀 Cómo Ejecutar

### Opción 1: Script de lanzamiento (Recomendado)

```bash
./run_web.sh
```

### Opción 2: Comando directo

```bash
source venv/bin/activate
python app/web_app.py
```

## 🌐 Acceder a la Aplicación

Una vez ejecutado, la aplicación estará disponible en:

- **Acceso local**: http://localhost:7860
- **Desde red local**: http://TU_IP:7860

El script mostrará la IP exacta al iniciar.

## 📱 Uso de la Aplicación

1. **Selecciona un modelo**
   - Por defecto viene seleccionado "SVM + HOG (97.3%)" - el mejor modelo
   - Puedes cambiar a otros modelos para comparar resultados

2. **Carga una imagen**
   - **Upload**: Sube una imagen desde tu computadora
   - **Webcam**: Toma una foto en tiempo real
   - **Clipboard**: Pega una imagen copiada

3. **Clasifica**
   - Haz clic en "🔍 Clasificar Imagen"
   - Ve el resultado: "🚗 PLACA DETECTADA" o "❌ NO ES PLACA"
   - Revisa el nivel de confianza (0-100%)
   - Observa la imagen procesada (128x128 escala de grises)

4. **Compara modelos**
   - Las métricas de cada modelo se muestran al seleccionarlo
   - Prueba la misma imagen con diferentes modelos
   - Compara precisión y confianza

## 📊 Modelos Disponibles

### 🏆 Mejor Rendimiento

1. **SVM + HOG** - 97.3% F1-Score (Recomendado)
2. **Neural Network + HOG** - 96.2% F1-Score
3. **Random Forest + HOG** - 95.7% F1-Score

### 📉 Rendimiento Moderado

4. **Random Forest + BRISK** - 80.6% F1-Score
5. **SVM + BRISK** - 78.8% F1-Score
6. **Neural Network + BRISK** - 75.1% F1-Score

## 🎨 Capturas de Pantalla

La interfaz incluye:

- **Panel izquierdo**: Entrada de imagen y selector de modelo
- **Panel derecho**: Resultados y visualizaciones
- **Sección inferior**: Información técnica y ejemplos

## 🔧 Configuración Avanzada

Para personalizar la aplicación, edita `app/web_app.py`:

```python
demo.launch(
    server_name="0.0.0.0",  # IP del servidor
    server_port=7860,        # Puerto (cambiar si está ocupado)
    share=True,              # True para link público temporal
    show_error=True          # Mostrar errores en la interfaz
)
```

### Opciones útiles:

- **`share=True`**: Genera un link público temporal de Gradio (útil para compartir)
- **`server_port`**: Cambia el puerto si 7860 está ocupado
- **`auth=("user", "pass")`**: Añade autenticación básica

## 🛠️ Solución de Problemas

### Error: "Address already in use"

```bash
# El puerto 7860 está ocupado, cambia el puerto en web_app.py
# O mata el proceso que lo usa:
lsof -ti:7860 | xargs kill -9
```

### Error: "No module named 'gradio'"

```bash
source venv/bin/activate
pip install gradio
```

### La aplicación no carga

```bash
# Verifica que estés en el directorio correcto
cd /home/rickcontreras/proyectos/car-plate-classifier

# Verifica que el venv esté activado
which python  # Debe mostrar la ruta del venv
```

## 📈 Ventajas vs Tkinter

| Característica | Gradio | Tkinter |
|---------------|--------|---------|
| Interfaz web | ✅ | ❌ |
| Acceso remoto | ✅ | ❌ |
| Responsive | ✅ | ❌ |
| Fácil deployment | ✅ | ❌ |
| Soporte webcam | ✅ | ❌ |
| Share público | ✅ | ❌ |
| Instalación | Pip | Sistema |

## 🌟 Próximos Pasos

1. **Deployment en producción**:
   - Hugging Face Spaces (gratis)
   - Docker container
   - Cloud (AWS, GCP, Azure)

2. **Mejoras posibles**:
   - Batch processing de imágenes
   - Historial de clasificaciones
   - API REST endpoint
   - Dashboard de estadísticas

## 📚 Documentación Adicional

- [Gradio Documentation](https://www.gradio.app/docs)
- [Gradio Gallery](https://www.gradio.app/gallery) - Ejemplos

## 💡 Tips de Uso

- **Mejor modelo**: Usa "SVM + HOG" para máxima precisión
- **Imágenes claras**: Mejor rendimiento con imágenes nítidas
- **Tamaño**: Las imágenes se redimensionan automáticamente a 128x128
- **Formato**: Acepta JPG, PNG, BMP y otros formatos comunes

---

🎓 **Desarrollado como parte del proyecto Car Plate Classifier**
📅 **Octubre 2025**
