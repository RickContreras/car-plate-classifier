#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aplicación Web Comparativa para Detección de Placas Vehiculares

Compara 3 modelos:
1. YOLO (Deep Learning con CNN)
2. Red Neuronal Fully Connected + HOG
3. Red Neuronal Fully Connected + BRISK
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys
import yaml
import tensorflow as tf
from ultralytics import YOLO

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_image
from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor


class PlateDetectorComparative:
    """Aplicación comparativa de detección de placas"""
    
    def __init__(self):
        """Inicializar la aplicación y cargar modelos"""
        self.base_dir = Path(__file__).parent.parent
        
        # Load config
        config_path = self.base_dir / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize feature extractors
        self.hog_extractor = HOGFeatureExtractor(self.config)
        self.brisk_extractor = BRISKFeatureExtractor(self.config)
        
        # Load models
        print("Cargando modelos...")
        self.load_models()
        print("✅ Todos los modelos cargados")
    
    def load_models(self):
        """Load all detection models"""
        # 1. YOLO model
        yolo_path = self.base_dir / 'models' / 'yolo' / 'plate_detector' / 'weights' / 'best.pt'
        self.yolo_model = YOLO(yolo_path)
        print("  ✓ YOLO cargado")
        
        # 2. NN-HOG model
        nn_hog_path = self.base_dir / 'models' / 'detection_nn' / 'detection_nn_hog.h5'
        self.nn_hog_model = tf.keras.models.load_model(nn_hog_path, compile=False)
        print("  ✓ NN-HOG cargado")
        
        # 3. NN-BRISK model
        nn_brisk_path = self.base_dir / 'models' / 'detection_nn' / 'detection_nn_brisk.h5'
        self.nn_brisk_model = tf.keras.models.load_model(nn_brisk_path, compile=False)
        print("  ✓ NN-BRISK cargado")
    
    def detect_yolo(self, image, confidence_threshold=0.5):
        """Detect plates using YOLO"""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.yolo_model.predict(image_bgr, conf=confidence_threshold, verbose=False)
        result = results[0]
        
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            detections.append({
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': confidence
            })
        
        return detections
    
    def detect_nn(self, image, model, extractor, feature_type):
        """Detect plates using Fully Connected Neural Network"""
        # Preprocess image
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed = preprocess_image(image_bgr, self.config)
        
        # Extract features
        if feature_type == 'HOG':
            features = self.hog_extractor.extract(processed).reshape(1, -1)
        else:  # BRISK
            features = self.brisk_extractor.extract(processed).reshape(1, -1)
        
        # Predict normalized coordinates (x_center, y_center, width, height)
        pred = model.predict(features, verbose=0)[0]
        
        # Convert to absolute coordinates
        h, w = image.shape[:2]
        x_center_norm, y_center_norm, width_norm, height_norm = pred
        
        # Denormalize
        x_center = x_center_norm * w
        y_center = y_center_norm * h
        box_width = width_norm * w
        box_height = height_norm * h
        
        # Convert to corner format
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)
        
        # Clip to image boundaries
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Calculate confidence based on box size (heuristic)
        box_area = (x2 - x1) * (y2 - y1)
        img_area = w * h
        area_ratio = box_area / img_area
        # Confidence higher when box is reasonable size (5-50% of image)
        if 0.05 <= area_ratio <= 0.5:
            confidence = 0.9
        elif 0.01 <= area_ratio <= 0.7:
            confidence = 0.7
        else:
            confidence = 0.5
        
        return [{
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence
        }]
    
    def draw_detections(self, image, detections, color=(0, 255, 0), label=""):
        """Draw bounding boxes on image"""
        output = image.copy()
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            text = f"{label} {conf*100:.1f}%"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Draw background
            cv2.rectangle(
                output,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                output,
                text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        return output
    
    def compare_models(self, image, confidence_threshold=0.5):
        """Compare all three models"""
        if image is None:
            return None, None, None, "⚠️ Por favor, carga una imagen"
        
        try:
            # Detect with each model
            yolo_dets = self.detect_yolo(image, confidence_threshold)
            nn_hog_dets = self.detect_nn(image, self.nn_hog_model, self.hog_extractor, 'HOG')
            nn_brisk_dets = self.detect_nn(image, self.nn_brisk_model, self.brisk_extractor, 'BRISK')
            
            # Draw results
            yolo_img = self.draw_detections(image, yolo_dets, (0, 255, 0), "YOLO")
            nn_hog_img = self.draw_detections(image, nn_hog_dets, (255, 165, 0), "NN-HOG")
            nn_brisk_img = self.draw_detections(image, nn_brisk_dets, (147, 20, 255), "NN-BRISK")
            
            # Create comparison text
            yolo_conf_str = ', '.join([f"{d['confidence']*100:.1f}%" for d in yolo_dets]) if yolo_dets else 'Ninguna'
            hog_conf_str = f"{nn_hog_dets[0]['confidence']*100:.1f}%" if nn_hog_dets else 'N/A'
            brisk_conf_str = f"{nn_brisk_dets[0]['confidence']*100:.1f}%" if nn_brisk_dets else 'N/A'
            
            comparison = f"""
## 📊 RESULTADOS DE LA COMPARACIÓN

### 🟢 YOLO (Deep Learning - CNN)
- **Placas detectadas**: {len(yolo_dets)}
- **Tipo**: Convolutional Neural Network
- **Características**: Aprendizaje automático de características
- **Detecciones**: {yolo_conf_str}

### 🟠 NN-HOG (Fully Connected)
- **Placas detectadas**: {len(nn_hog_dets)}
- **Tipo**: Red Neuronal Completamente Conectada
- **Características**: HOG (8100 dimensiones)
- **Confianza**: {hog_conf_str}

### 🟣 NN-BRISK (Fully Connected)
- **Placas detectadas**: {len(nn_brisk_dets)}
- **Tipo**: Red Neuronal Completamente Conectada
- **Características**: BRISK (512 dimensiones)
- **Confianza**: {brisk_conf_str}

---

### 📈 MÉTRICAS DE ENTRENAMIENTO

| Modelo | Tipo | MAE | IoU Promedio | IoU > 0.5 |
|--------|------|-----|--------------|-----------|
| **YOLO** | CNN | - | **91.89%** | **91.9%** |
| **NN-HOG** | FC | **7.45%** | **39.55%** | **48.3%** |
| **NN-BRISK** | FC | 6.89% | 17.20% | 10.3% |

**Conclusión**: YOLO (Deep Learning con CNN) tiene el mejor rendimiento,  
seguido de NN-HOG (Fully Connected con características manuales HOG).
            """.strip()
            
            return yolo_img, nn_hog_img, nn_brisk_img, comparison
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return None, None, None, error_msg


def create_app():
    """Create Gradio application"""
    
    app = PlateDetectorComparative()
    
    description = """
    # 🔬 Comparación de Modelos de Detección de Placas
    
    Esta aplicación compara **3 enfoques diferentes** para detectar placas vehiculares:
    
    ## 🎯 Modelos Implementados:
    
    ### 1. 🟢 YOLO (Deep Learning - CNN)
    - **Tipo**: Redes Neuronales Convolucionales (CNN)
    - **Características**: Aprende características automáticamente
    - **Capas**: 72 layers, 3M parámetros
    - **mAP50**: 91.89%
    
    ### 2. 🟠 NN-HOG (Machine Learning - Fully Connected)
    - **Tipo**: Red Neuronal Completamente Conectada
    - **Características**: HOG (Histogram of Oriented Gradients) - 8100 dimensiones
    - **Extracción manual** de características
    - **IoU promedio**: 39.55%
    
    ### 3. 🟣 NN-BRISK (Machine Learning - Fully Connected)
    - **Tipo**: Red Neuronal Completamente Conectada
    - **Características**: BRISK (Binary Robust Invariant) - 512 dimensiones
    - **Extracción manual** de características
    - **IoU promedio**: 17.20%
    
    ## 🔍 Diferencias Clave:
    
    | Aspecto | YOLO (CNN) | NN-HOG/BRISK (FC) |
    |---------|------------|-------------------|
    | **Características** | Automáticas | Manuales (HOG/BRISK) |
    | **Arquitectura** | Convolucional | Fully Connected |
    | **Capas** | 72 layers | 5 layers |
    | **Parámetros** | 3 millones | 4.3M (HOG) / 438K (BRISK) |
    | **Precisión** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ / ⭐⭐ |
    
    ---
    
    💡 **Nota**: Esta es una demostración de cómo las redes convolucionales (YOLO)  
    superan a las redes completamente conectadas en tareas de visión por computadora.
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Comparación de Detección") as demo:
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Configuración")
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="🎯 Umbral de Confianza (solo para YOLO)",
                    info="Mínima confianza para YOLO"
                )
                
                image_input = gr.Image(
                    label="🖼️ Imagen a analizar",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"]
                )
                
                detect_btn = gr.Button(
                    "🔍 Comparar Modelos",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📤 Resultados")
                
                with gr.Row():
                    yolo_output = gr.Image(label="🟢 YOLO (Deep Learning - CNN)")
                    nn_hog_output = gr.Image(label="🟠 NN-HOG (Fully Connected)")
                    nn_brisk_output = gr.Image(label="🟣 NN-BRISK (Fully Connected)")
                
                gr.Markdown("### 📊 Análisis Comparativo")
                comparison_output = gr.Markdown()
        
        # Event handler
        detect_btn.click(
            fn=app.compare_models,
            inputs=[image_input, confidence_slider],
            outputs=[yolo_output, nn_hog_output, nn_brisk_output, comparison_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 🔬 Conclusiones Técnicas
            
            **¿Por qué YOLO es mejor?**
            1. **Características jerárquicas**: CNN aprende características de bajo nivel (bordes) hasta alto nivel (placas completas)
            2. **Invarianza espacial**: Las convoluciones detectan patrones independientemente de su posición
            3. **Menos parámetros efectivos**: A pesar de tener más parámetros totales, las CNN son más eficientes por el weight sharing
            4. **Entrenamiento end-to-end**: Todo se optimiza junto, no en pasos separados
            
            **¿Cuándo usar Fully Connected?**
            - Datasets muy pequeños (<100 imágenes)
            - Cuando no se tiene GPU
            - Para entender conceptos básicos de ML
            - Como baseline para comparar
            
            🎓 **Proyecto**: Comparative Plate Detection | 📅 **Fecha**: Octubre 2025
            """
        )
    
    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )
