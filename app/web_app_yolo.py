#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aplicación Web con Gradio para Detección de Placas Vehiculares usando YOLO
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class PlateDetectorApp:
    """Aplicación web para detección de placas con YOLO"""
    
    def __init__(self):
        """Inicializar la aplicación"""
        self.base_dir = Path(__file__).parent.parent
        self.model_path = self.base_dir / 'models' / 'yolo' / 'plate_detector' / 'weights' / 'best.pt'
        
        # Cargar modelo YOLO
        print(f"Cargando modelo YOLO desde: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("✅ Modelo YOLO cargado exitosamente")
    
    def detect_plates(self, image, confidence_threshold=0.5, draw_boxes=True):
        """
        Detectar placas en una imagen usando YOLO
        
        Args:
            image: Imagen de numpy array (RGB)
            confidence_threshold: Umbral de confianza mínimo (0.0 - 1.0)
            draw_boxes: Si True, dibuja los bounding boxes
        
        Returns:
            tuple: (imagen_con_detecciones, resultado_texto, métricas)
        """
        if image is None:
            return None, "⚠️ Por favor, carga una imagen", ""
        
        try:
            # Convertir RGB a BGR para OpenCV (YOLO espera BGR)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Realizar detección
            results = self.model.predict(
                image_bgr,
                conf=confidence_threshold,
                verbose=False
            )
            
            # Obtener el primer resultado (solo procesamos una imagen)
            result = results[0]
            
            # Número de placas detectadas
            num_plates = len(result.boxes)
            
            # Crear imagen de salida
            output_image = image.copy()
            
            # Información de detecciones
            detections_info = []
            
            if num_plates > 0:
                for idx, box in enumerate(result.boxes):
                    # Obtener coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Convertir a enteros
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calcular dimensiones
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Guardar información
                    detections_info.append({
                        'id': idx + 1,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'width': width,
                        'height': height
                    })
                    
                    if draw_boxes:
                        # Color según confianza (verde si alta, amarillo si media)
                        if confidence >= 0.8:
                            color = (0, 255, 0)  # Verde (RGB)
                        elif confidence >= 0.6:
                            color = (255, 255, 0)  # Amarillo
                        else:
                            color = (255, 165, 0)  # Naranja
                        
                        # Dibujar rectángulo
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)
                        
                        # Preparar texto
                        label = f"Placa #{idx+1}: {confidence*100:.1f}%"
                        
                        # Calcular tamaño del texto
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Dibujar fondo para el texto
                        cv2.rectangle(
                            output_image,
                            (x1, y1 - text_height - 10),
                            (x1 + text_width + 10, y1),
                            color,
                            -1
                        )
                        
                        # Dibujar texto
                        cv2.putText(
                            output_image,
                            label,
                            (x1 + 5, y1 - 5),
                            font,
                            font_scale,
                            (0, 0, 0),  # Negro
                            thickness
                        )
            
            # Crear texto de resultado
            if num_plates == 0:
                result_text = "❌ NO SE DETECTARON PLACAS"
                metrics_text = "No se encontraron placas en la imagen.\nIntenta con otra imagen o ajusta el umbral de confianza."
            elif num_plates == 1:
                conf = detections_info[0]['confidence']
                result_text = f"✅ 1 PLACA DETECTADA ({conf*100:.1f}% confianza)"
                x1, y1, x2, y2 = detections_info[0]['bbox']
                w, h = detections_info[0]['width'], detections_info[0]['height']
                metrics_text = f"""
📊 **Información de la Detección:**

• **Ubicación**: ({x1}, {y1}) → ({x2}, {y2})
• **Dimensiones**: {w} × {h} píxeles
• **Confianza**: {conf*100:.2f}%
• **Modelo**: YOLOv8n (nano)
• **mAP50**: 91.89%
                """.strip()
            else:
                result_text = f"✅ {num_plates} PLACAS DETECTADAS"
                avg_conf = np.mean([d['confidence'] for d in detections_info]) * 100
                metrics_text = f"**Total de placas**: {num_plates}\n**Confianza promedio**: {avg_conf:.1f}%\n\n"
                
                for det in detections_info:
                    x1, y1, x2, y2 = det['bbox']
                    w, h = det['width'], det['height']
                    conf = det['confidence'] * 100
                    metrics_text += f"\n**Placa #{det['id']}:**\n"
                    metrics_text += f"  • Ubicación: ({x1}, {y1}) → ({x2}, {y2})\n"
                    metrics_text += f"  • Dimensiones: {w} × {h} px\n"
                    metrics_text += f"  • Confianza: {conf:.2f}%\n"
            
            return output_image, result_text, metrics_text
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return None, f"❌ Error: {str(e)}", f"Detalles del error:\n{error_details}"


def create_app():
    """Crear aplicación Gradio"""
    
    app = PlateDetectorApp()
    
    # Descripción
    description = """
    # 🚗 Detector de Placas Vehiculares con YOLO
    
    Esta aplicación utiliza **Deep Learning (YOLOv8)** para **detectar y localizar** placas vehiculares en imágenes.
    
    ## 🎯 Características del Sistema:
    - ✅ **Modelo**: YOLOv8n (nano) - Rápido y eficiente
    - ✅ **Precisión**: 91.89% mAP50
    - ✅ **Dataset**: 433 imágenes (346 train / 87 val)
    - ✅ **Detección en tiempo real**: ~92ms por imagen
    - ✅ **Localización**: Dibuja bounding boxes sobre las placas detectadas
    
    ## 🔍 Ventajas sobre Clasificación:
    - ✨ **Detecta WHERE** (dónde está la placa), no solo IF (si hay placa)
    - ✨ **Múltiples placas**: Puede detectar varias placas en una misma imagen
    - ✨ **Coordenadas precisas**: Proporciona ubicación exacta (x, y, width, height)
    - ✨ **Confianza por detección**: Cada placa tiene su nivel de confianza
    
    ## 🎯 Cómo usar:
    1. Ajusta el umbral de confianza (recomendado: 0.5)
    2. Carga una imagen o toma una foto
    3. ¡Ve las placas detectadas con sus bounding boxes!
    """
    
    # Crear interfaz
    with gr.Blocks(theme=gr.themes.Soft(), title="Detector de Placas YOLO") as demo:
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Entrada")
                
                # Umbral de confianza
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="🎯 Umbral de Confianza",
                    info="Mínima confianza para considerar una detección (0.1 = 10%, 1.0 = 100%)"
                )
                
                # Input de imagen
                image_input = gr.Image(
                    label="🖼️ Imagen a analizar",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"]
                )
                
                # Checkbox para mostrar/ocultar boxes
                show_boxes = gr.Checkbox(
                    label="📦 Mostrar bounding boxes",
                    value=True,
                    info="Si está activado, dibuja rectángulos sobre las placas detectadas"
                )
                
                # Botón de detección
                detect_btn = gr.Button(
                    "🔍 Detectar Placas",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Resultado")
                
                # Resultado de detección
                result_output = gr.Textbox(
                    label="🎯 Resultado",
                    lines=1,
                    scale=1
                )
                
                # Imagen con detecciones
                output_image = gr.Image(
                    label="🔍 Imagen con Detecciones",
                    type="numpy"
                )
                
                # Métricas de detección
                gr.Markdown("### 📊 Información de Detecciones")
                metrics_output = gr.Markdown(
                    value="Carga una imagen y presiona 'Detectar Placas' para ver los resultados."
                )
        
        # Ejemplos
        gr.Markdown("### 💡 Información del Modelo")
        gr.Markdown(
            """
            **YOLOv8n (You Only Look Once - Version 8 Nano)**
            
            - **Arquitectura**: Deep Learning con Redes Neuronales Convolucionales (CNN)
            - **Capas**: 72 layers
            - **Parámetros**: 3,005,843 parámetros entrenables
            - **Velocidad de inferencia**: ~92.5ms por imagen (CPU)
            - **Entrenamiento**: 42 épocas con early stopping
            - **Métricas de validación**:
              - **Precision**: 91.70%
              - **Recall**: 90.80%
              - **mAP50**: 91.89%
              - **mAP50-95**: 59.08%
            
            🎓 **Diferencia con modelos anteriores**:
            - Los modelos anteriores (SVM, Random Forest) solo **clasificaban** (PLACA: SÍ/NO)
            - YOLO **detecta y localiza** (DÓNDE está la placa con coordenadas precisas)
            - Usa Deep Learning en lugar de Machine Learning tradicional
            """
        )
        
        # Evento de detección
        detect_btn.click(
            fn=app.detect_plates,
            inputs=[image_input, confidence_slider, show_boxes],
            outputs=[output_image, result_output, metrics_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 🛠️ Información Técnica
            - **Framework**: Ultralytics YOLOv8
            - **Input size**: 640×640 píxeles
            - **Output**: Bounding boxes (x1, y1, x2, y2) + confianza
            - **Classes**: 1 clase ('plate')
            - **Dataset format**: YOLO format (normalized coordinates)
            
            🔬 **Proyecto**: Car Plate Detector with YOLO | 📅 **Fecha**: Octubre 2025
            """
        )
    
    return demo


if __name__ == "__main__":
    # Crear y lanzar aplicación
    demo = create_app()
    
    # Lanzar con configuración
    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde red local
        server_port=7861,  # Puerto diferente al clasificador original
        share=False,  # Cambiar a True para obtener link público temporal
        show_error=True
    )
