"""
Aplicación Web de Detección de Placas Vehiculares
Usa modelos HOG y BRISK para detectar placas en imágenes de autos
"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Usar backend sin GUI para matplotlib

import gradio as gr
import cv2
import numpy as np
from tensorflow import keras
import time
from src.features.hog import HOGFeatureExtractor
from src.features.brisk import BRISKFeatureExtractor
from src.data.utils import denormalize_bbox as denorm_bbox

# Cargar modelos (solo una vez al inicio)
print("🔄 Cargando modelos...")
hog_model = keras.models.load_model('models/detection_hog_best.h5', compile=False)
brisk_model = keras.models.load_model('models/detection_brisk_best.h5', compile=False)
print("✅ Modelos cargados!")

# Inicializar extractores de features
hog_extractor = HOGFeatureExtractor()
brisk_extractor = BRISKFeatureExtractor()

def draw_bbox(image, bbox, label, color, confidence=None):
    """Dibujar bounding box en la imagen"""
    img = image.copy()
    x1, y1, x2, y2 = bbox
    
    # Dibujar rectángulo
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    
    # Preparar texto
    text = label
    if confidence is not None:
        text += f" ({confidence:.1f}%)"
    
    # Fondo para el texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Dibujar fondo del texto
    cv2.rectangle(img, 
                 (x1, y1 - text_size[1] - 10),
                 (x1 + text_size[0] + 10, y1),
                 color, -1)
    
    # Dibujar texto
    cv2.putText(img, text, (x1 + 5, y1 - 5),
                font, font_scale, (255, 255, 255), thickness)
    
    return img

def calculate_confidence(bbox):
    """Calcular pseudo-confianza basada en el tamaño del bbox"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    
    # Áreas razonables para placas (0.01 - 0.15 del área total)
    if 0.01 <= area <= 0.15:
        confidence = min(100, 50 + (area / 0.15) * 50)
    else:
        confidence = max(30, 100 - abs(area - 0.08) * 500)
    
    return min(100, max(30, confidence))

def detect_plate(image, model_choice):
    """
    Detectar placa en la imagen usando el modelo seleccionado
    
    Args:
        image: imagen numpy array (RGB)
        model_choice: "HOG", "BRISK", o "Ambos"
    
    Returns:
        imagen con bounding boxes dibujados, tiempo de procesamiento, detalles
    """
    if image is None:
        return None, "⚠️ Por favor, sube una imagen", ""
    
    # Convertir RGB a BGR (OpenCV usa BGR)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results_text = "## 📊 Resultados de Detección\n\n"
    details_text = ""
    
    # Crear imagen para mostrar
    result_img = image.copy()
    
    if model_choice in ["HOG", "Ambos"]:
        # Detección con HOG
        start_time = time.time()
        
        try:
            # Extraer features
            hog_features = hog_extractor.extract(img_bgr)
            hog_features = hog_features.reshape(1, -1)
            
            # Predecir
            hog_bbox_norm = hog_model.predict(hog_features, verbose=0)[0]
            h, w = img_bgr.shape[:2]
            hog_bbox = denorm_bbox(hog_bbox_norm, w, h)
            
            hog_time = (time.time() - start_time) * 1000
            hog_confidence = calculate_confidence(hog_bbox_norm)
            
            # Dibujar bbox HOG en verde
            result_img = draw_bbox(result_img, hog_bbox, "HOG", 
                                  (0, 255, 0), hog_confidence)
            
            results_text += f"### 🟢 Modelo HOG\n"
            results_text += f"- **Tiempo**: {hog_time:.1f} ms\n"
            results_text += f"- **Confianza**: {hog_confidence:.1f}%\n"
            results_text += f"- **Bbox**: [{hog_bbox[0]}, {hog_bbox[1]}, {hog_bbox[2]}, {hog_bbox[3]}]\n"
            results_text += f"- **Área detectada**: {(hog_bbox_norm[2]-hog_bbox_norm[0])*(hog_bbox_norm[3]-hog_bbox_norm[1])*100:.2f}%\n\n"
            
            details_text += f"🟢 **HOG**: {hog_time:.1f}ms | Confianza: {hog_confidence:.1f}%\n"
            
        except Exception as e:
            results_text += f"### ❌ Error en modelo HOG\n{str(e)}\n\n"
            details_text += f"❌ **HOG**: Error\n"
    
    if model_choice in ["BRISK", "Ambos"]:
        # Detección con BRISK
        start_time = time.time()
        
        try:
            # Extraer features
            brisk_features = brisk_extractor.extract(img_bgr)
            brisk_features = brisk_features.reshape(1, -1)
            
            # Predecir
            brisk_bbox_norm = brisk_model.predict(brisk_features, verbose=0)[0]
            h, w = img_bgr.shape[:2]
            brisk_bbox = denorm_bbox(brisk_bbox_norm, w, h)
            
            brisk_time = (time.time() - start_time) * 1000
            brisk_confidence = calculate_confidence(brisk_bbox_norm)
            
            # Dibujar bbox BRISK en azul
            color = (255, 165, 0) if model_choice == "Ambos" else (0, 0, 255)
            result_img = draw_bbox(result_img, brisk_bbox, "BRISK", 
                                  color, brisk_confidence)
            
            results_text += f"### 🔵 Modelo BRISK\n"
            results_text += f"- **Tiempo**: {brisk_time:.1f} ms\n"
            results_text += f"- **Confianza**: {brisk_confidence:.1f}%\n"
            results_text += f"- **Bbox**: [{brisk_bbox[0]}, {brisk_bbox[1]}, {brisk_bbox[2]}, {brisk_bbox[3]}]\n"
            results_text += f"- **Área detectada**: {(brisk_bbox_norm[2]-brisk_bbox_norm[0])*(brisk_bbox_norm[3]-brisk_bbox_norm[1])*100:.2f}%\n\n"
            
            details_text += f"🔵 **BRISK**: {brisk_time:.1f}ms | Confianza: {brisk_confidence:.1f}%\n"
            
        except Exception as e:
            results_text += f"### ❌ Error en modelo BRISK\n{str(e)}\n\n"
            details_text += f"❌ **BRISK**: Error\n"
    
    if model_choice == "Ambos":
        results_text += "---\n"
        results_text += "### 💡 Interpretación\n"
        results_text += "- **Verde (HOG)**: Más preciso, más lento\n"
        results_text += "- **Naranja (BRISK)**: Más rápido, menos preciso\n"
        results_text += "- Si ambos boxes coinciden → Alta confianza\n"
        results_text += "- Si difieren mucho → Revisar imagen\n"
    
    return result_img, results_text, details_text

# Crear interfaz Gradio
with gr.Blocks(title="🚗 Detector de Placas Vehiculares", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🚗 Detector de Placas Vehiculares
    ### Detección automática de placas usando Deep Learning + Feature Extraction
    
    Sube una imagen de un auto y selecciona el modelo para detectar la placa automáticamente.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input
            input_image = gr.Image(
                label="📸 Sube una imagen de un auto",
                type="numpy",
                height=400
            )
            
            model_selector = gr.Radio(
                choices=["HOG", "BRISK", "Ambos"],
                value="Ambos",
                label="🎯 Selecciona el modelo",
                info="HOG es más preciso pero lento, BRISK es rápido pero menos preciso"
            )
            
            detect_btn = gr.Button("🔍 Detectar Placa", variant="primary", size="lg")
            
            gr.Markdown("""
            ### 📝 Información de los Modelos
            
            **🟢 HOG (Histogram of Oriented Gradients)**
            - Precisión: ★★★★☆ (31% @ IoU 0.5)
            - Velocidad: ★★☆☆☆ (~9 FPS)
            - Features: 42,849 dimensiones
            
            **🔵 BRISK (Binary Robust Invariant Scalable Keypoints)**
            - Precisión: ★★★☆☆ (21% @ IoU 0.5)
            - Velocidad: ★★★★☆ (~11 FPS)
            - Features: 32,768 dimensiones
            
            **💡 Consejo**: Usa "Ambos" para comparar resultados
            """)
        
        with gr.Column(scale=1):
            # Output
            output_image = gr.Image(
                label="🎯 Detección Resultante",
                height=400
            )
            
            output_text = gr.Markdown(label="📊 Resultados")
            
            quick_info = gr.Textbox(
                label="⚡ Resumen Rápido",
                lines=3,
                interactive=False
            )
    
    # Ejemplos
    gr.Markdown("### 🖼️ Ejemplos de Prueba")
    gr.Markdown("Haz clic en un ejemplo para cargarlo automáticamente:")
    
    gr.Examples(
        examples=[
            ["data/raw/Cars0.png", "Ambos"],
            ["data/raw/Cars1.png", "HOG"],
            ["data/raw/Cars2.png", "BRISK"],
        ],
        inputs=[input_image, model_selector],
        outputs=[output_image, output_text, quick_info],
        fn=detect_plate,
        cache_examples=False,
        label="Ejemplos"
    )
    
    # Event handler
    detect_btn.click(
        fn=detect_plate,
        inputs=[input_image, model_selector],
        outputs=[output_image, output_text, quick_info]
    )
    
    gr.Markdown("""
    ---
    ### 🔧 Detalles Técnicos
    
    - **Dataset**: 433 imágenes de autos con placas anotadas
    - **Arquitectura**: Fully Connected Network (5 capas)
    - **Entrenamiento**: ~75-97 epochs con Early Stopping
    - **Framework**: TensorFlow/Keras + OpenCV + scikit-image
    
    ### 📊 Métricas de Evaluación
    
    | Modelo | IoU Promedio | Precisión @0.5 | Velocidad | Tamaño |
    |--------|--------------|----------------|-----------|--------|
    | HOG    | 30%          | 31%            | 112 ms    | 254 MB |
    | BRISK  | 25%          | 21%            | 93 ms     | 195 MB |
    
    ### 🎯 Leyenda de Colores
    
    - 🟢 **Verde**: Detección del modelo HOG
    - 🔵 **Azul**: Detección del modelo BRISK (modo individual)
    - 🟠 **Naranja**: Detección del modelo BRISK (modo comparación)
    
    ---
    **Proyecto**: Car Plate Detection | **Framework**: Gradio v5 | **Año**: 2025
    """)

# Lanzar aplicación
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Iniciando aplicación web de detección de placas...")
    print("="*70 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",  # Solo accesible localmente
        server_port=7861,          # Puerto alternativo
        share=False,               # Cambiar a True para generar URL pública
        show_error=True,
        quiet=False
    )
