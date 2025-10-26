#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Aplicación Web con Gradio para Clasificador de Placas Vehiculares
"""

import gradio as gr
import cv2
import numpy as np
import joblib
import yaml
import tensorflow as tf
from pathlib import Path
import sys

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_image
from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor


class PlateClassifierApp:
    """Aplicación web para clasificación de placas"""
    
    def __init__(self):
        """Inicializar la aplicación"""
        self.base_dir = Path(__file__).parent.parent
        self.models_dir = self.base_dir / 'models'
        self.config_path = self.base_dir / 'config' / 'config.yaml'
        
        # Cargar configuración
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Inicializar extractores de características
        self.hog_extractor = HOGFeatureExtractor(self.config)
        self.brisk_extractor = BRISKFeatureExtractor(self.config)
        
        # Almacenar modelos cargados
        self.loaded_models = {}
        
        # Información de modelos disponibles
        self.models_info = {
            'SVM + HOG (97.3%)': {
                'path': 'svm_hog.pkl',
                'extractor': 'hog',
                'type': 'sklearn',
                'metrics': '🏆 MEJOR MODELO\nAccuracy: 97.24%\nPrecision: 98.90%\nRecall: 95.74%\nF1-Score: 97.30%'
            },
            'Neural Network + HOG (96.2%)': {
                'path': 'neural_network_hog.h5',
                'extractor': 'hog',
                'type': 'keras',
                'metrics': 'Accuracy: 96.13%\nPrecision: 97.80%\nRecall: 94.68%\nF1-Score: 96.22%'
            },
            'Random Forest + HOG (95.7%)': {
                'path': 'random_forest_hog.pkl',
                'extractor': 'hog',
                'type': 'sklearn',
                'metrics': 'Accuracy: 95.58%\nPrecision: 96.74%\nRecall: 94.68%\nF1-Score: 95.70%'
            },
            'Random Forest + BRISK (80.6%)': {
                'path': 'random_forest_brisk.pkl',
                'extractor': 'brisk',
                'type': 'sklearn',
                'metrics': 'Accuracy: 79.56%\nPrecision: 79.38%\nRecall: 81.91%\nF1-Score: 80.63%'
            },
            'SVM + BRISK (78.8%)': {
                'path': 'svm_brisk.pkl',
                'extractor': 'brisk',
                'type': 'sklearn',
                'metrics': 'Accuracy: 76.80%\nPrecision: 75.00%\nRecall: 82.98%\nF1-Score: 78.79%'
            },
            'Neural Network + BRISK (75.1%)': {
                'path': 'neural_network_brisk.h5',
                'extractor': 'brisk',
                'type': 'keras',
                'metrics': 'Accuracy: 70.72%\nPrecision: 67.23%\nRecall: 85.11%\nF1-Score: 75.12%'
            }
        }
    
    def load_model(self, model_name):
        """Cargar modelo si no está cargado"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_info = self.models_info[model_name]
        model_path = self.models_dir / model_info['path']
        
        try:
            if model_info['type'] == 'sklearn':
                # Usar joblib en lugar de pickle para modelos sklearn
                model = joblib.load(model_path)
            else:  # keras
                model = tf.keras.models.load_model(model_path)
            
            self.loaded_models[model_name] = model
            return model
        except Exception as e:
            raise Exception(f"Error al cargar modelo: {str(e)}")
    
    def classify_image(self, image, model_name):
        """
        Clasificar una imagen
        
        Args:
            image: Imagen de numpy array (RGB)
            model_name: Nombre del modelo a usar
        
        Returns:
            tuple: (resultado, confianza, imagen_procesada, métricas)
        """
        if image is None:
            return "⚠️ Por favor, carga una imagen", "", None, ""
        
        try:
            # Cargar modelo
            model = self.load_model(model_name)
            model_info = self.models_info[model_name]
            
            # Convertir RGB a BGR para OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Preprocesar imagen
            processed = preprocess_image(image_bgr, self.config)
            
            # Extraer características
            if model_info['extractor'] == 'hog':
                features = self.hog_extractor.extract(processed).reshape(1, -1)
            else:  # brisk
                features = self.brisk_extractor.extract(processed).reshape(1, -1)
            
            # Realizar predicción
            if model_info['type'] == 'sklearn':
                prediction = model.predict(features)[0]
                # Obtener probabilidades si está disponible
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    confidence = proba[int(prediction)] * 100
                else:
                    confidence = 100.0 if prediction == 1 else 0.0
            else:  # keras
                proba = model.predict(features, verbose=0)[0][0]
                prediction = 1 if proba > 0.5 else 0
                confidence = proba * 100 if prediction == 1 else (1 - proba) * 100
            
            # Resultado
            result_text = "🚗 PLACA DETECTADA" if prediction == 1 else "❌ NO ES PLACA"
            confidence_text = f"Confianza: {confidence:.2f}%"
            
            # Convertir imagen procesada de vuelta a uint8 para visualización
            # Si está normalizada (0-1), multiplicar por 255
            if processed.dtype == np.float32 or processed.dtype == np.float64:
                processed_display = (processed * 255).astype(np.uint8)
            else:
                processed_display = processed
            
            # Convertir imagen procesada a RGB para mostrar
            if len(processed_display.shape) == 2:  # Si es escala de grises
                processed_rgb = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = processed_display
            
            # Añadir borde de color según resultado
            border_color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
            processed_rgb = cv2.copyMakeBorder(
                processed_rgb, 5, 5, 5, 5, 
                cv2.BORDER_CONSTANT, value=border_color
            )
            
            return result_text, confidence_text, processed_rgb, model_info['metrics']
            
        except Exception as e:
            return f"❌ Error: {str(e)}", "", None, ""


def create_app():
    """Crear aplicación Gradio"""
    
    app = PlateClassifierApp()
    
    # Descripción
    description = """
    # 🚗 Clasificador de Placas Vehiculares
    
    Esta aplicación utiliza **Machine Learning** para detectar si una imagen contiene una placa vehicular.
    
    ## 📊 Características del Sistema:
    - ✅ **6 modelos entrenados** (SVM, Random Forest, Neural Network)
    - ✅ **2 tipos de características** (HOG, BRISK)
    - ✅ **Dataset**: 904 imágenes (433 originales + 471 procesadas)
    - ✅ **Mejor modelo**: SVM + HOG con **97.3% F1-Score**
    
    ## 🎯 Cómo usar:
    1. Selecciona un modelo (recomendado: SVM + HOG)
    2. Carga una imagen o toma una foto
    3. ¡Ve el resultado instantáneamente!
    """
    
    # Crear interfaz
    with gr.Blocks(theme=gr.themes.Soft(), title="Clasificador de Placas") as demo:
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Entrada")
                
                # Selector de modelo
                model_selector = gr.Dropdown(
                    choices=list(app.models_info.keys()),
                    value='SVM + HOG (97.3%)',
                    label="🤖 Modelo a usar",
                    info="Selecciona el modelo para clasificación"
                )
                
                # Input de imagen
                image_input = gr.Image(
                    label="🖼️ Imagen a clasificar",
                    type="numpy",
                    sources=["upload", "webcam", "clipboard"]
                )
                
                # Botón de clasificación
                classify_btn = gr.Button(
                    "🔍 Clasificar Imagen",
                    variant="primary",
                    size="lg"
                )
                
                # Métricas del modelo
                gr.Markdown("### 📊 Métricas del Modelo")
                metrics_output = gr.Textbox(
                    label="Rendimiento",
                    lines=5,
                    value=app.models_info['SVM + HOG (97.3%)']['metrics']
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Resultado")
                
                # Resultado de clasificación
                result_output = gr.Textbox(
                    label="🎯 Predicción",
                    lines=1,
                    text_align="center"
                )
                
                # Confianza
                confidence_output = gr.Textbox(
                    label="📈 Nivel de Confianza",
                    lines=1
                )
                
                # Imagen procesada
                processed_output = gr.Image(
                    label="🔍 Imagen Procesada (128x128 escala de grises)",
                    type="numpy"
                )
        
        # Ejemplos
        gr.Markdown("### 🖼️ Ejemplos de Uso")
        gr.Markdown(
            "Puedes probar con imágenes de placas vehiculares o imágenes que no contengan placas. "
            "El modelo ha sido entrenado con imágenes de 128x128 píxeles en escala de grises."
        )
        
        # Actualizar métricas cuando cambie el modelo
        model_selector.change(
            fn=lambda x: app.models_info[x]['metrics'],
            inputs=[model_selector],
            outputs=[metrics_output]
        )
        
        # Evento de clasificación
        classify_btn.click(
            fn=app.classify_image,
            inputs=[image_input, model_selector],
            outputs=[result_output, confidence_output, processed_output, metrics_output]
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            ### 🛠️ Información Técnica
            - **Preprocesamiento**: 128x128 píxeles, escala de grises, normalización
            - **HOG**: 8100 características (9 orientaciones, 8x8 píxeles/celda, 2x2 celdas/bloque)
            - **BRISK**: 512 características (threshold=30, octaves=3)
            - **Dataset split**: 80% entrenamiento, 20% prueba
            
            🔬 **Proyecto**: Car Plate Classifier | 📅 **Fecha**: Octubre 2025
            """
        )
    
    return demo


if __name__ == "__main__":
    # Crear y lanzar aplicación
    demo = create_app()
    
    # Lanzar con configuración
    demo.launch(
        server_name="0.0.0.0",  # Permite acceso desde red local
        server_port=7860,
        share=False,  # Cambiar a True para obtener link público temporal
        show_error=True
    )
