"""
Backbones (redes base) para extracción de características en RetinaNet.

Este módulo provee diferentes arquitecturas pre-entrenadas que extraen
features jerárquicas a múltiples escalas (C2, C3, C4, C5).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Tuple, Dict, Optional


class ResNetBackbone:
    """
    ResNet-50 como backbone para extracción de características.
    
    Extrae features a múltiples niveles de resolución que luego son
    usados por el Feature Pyramid Network (FPN) para detección multi-escala.
    
    Niveles de salida:
    - C2: stride 4 (alta resolución, detalles finos)
    - C3: stride 8
    - C4: stride 16
    - C5: stride 32 (baja resolución, features semánticos)
    
    Attributes:
        input_shape: Forma de entrada (height, width, channels)
        weights: Pesos pre-entrenados ('imagenet', None, o ruta)
        trainable: Si el backbone es entrenable o congelado
        base_model: Modelo ResNet50 de Keras
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        weights: str = 'imagenet',
        trainable: bool = False
    ):
        """
        Inicializar ResNet backbone.
        
        Args:
            input_shape: Forma de entrada (height, width, channels)
            weights: Pesos pre-entrenados ('imagenet' recomen dado)
            trainable: Si True, fine-tuning del backbone (más lento, mejor precisión)
        """
        self.input_shape = input_shape
        self.weights = weights
        self.trainable = trainable
        
        # Crear modelo base ResNet50
        self.base_model = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape
        )
        
        # Congelar capas si no es entrenable
        if not trainable:
            self.base_model.trainable = False
    
    def build(self) -> keras.Model:
        """
        Construir modelo de backbone con múltiples salidas.
        
        Returns:
            Modelo de Keras con salidas [C2, C3, C4, C5]
        """
        # Capas de salida para cada nivel de features
        output_layer_names = [
            'conv2_block3_out',   # C2: stride 4, shape (H/4, W/4, 256)
            'conv3_block4_out',   # C3: stride 8, shape (H/8, W/8, 512)
            'conv4_block6_out',   # C4: stride 16, shape (H/16, W/16, 1024)
            'conv5_block3_out',   # C5: stride 32, shape (H/32, W/32, 2048)
        ]
        
        # Extraer outputs de las capas especificadas
        outputs = [
            self.base_model.get_layer(name).output
            for name in output_layer_names
        ]
        
        # Crear modelo multi-output
        model = keras.Model(
            inputs=self.base_model.input,
            outputs=outputs,
            name='resnet50_backbone'
        )
        
        return model
    
    def get_config(self) -> Dict:
        """
        Obtener configuración del backbone.
        
        Returns:
            Diccionario con parámetros de configuración
        """
        return {
            'type': 'resnet50',
            'input_shape': self.input_shape,
            'weights': self.weights,
            'trainable': self.trainable
        }
    
    def __repr__(self) -> str:
        """Representación en string del backbone."""
        return (f"ResNetBackbone(input_shape={self.input_shape}, "
                f"weights='{self.weights}', trainable={self.trainable})")


class MobileNetBackbone:
    """
    MobileNetV2 como backbone ligero alternativo.
    
    Ideal para deployment en dispositivos con recursos limitados.
    Menor precisión que ResNet50 pero mucho más rápido.
    
    Attributes:
        input_shape: Forma de entrada
        alpha: Multiplicador de ancho (0.35, 0.5, 0.75, 1.0)
        weights: Pesos pre-entrenados
        trainable: Si es entrenable
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        alpha: float = 1.0,
        weights: str = 'imagenet',
        trainable: bool = False
    ):
        """
        Inicializar MobileNet backbone.
        
        Args:
            input_shape: Forma de entrada
            alpha: Factor de escala (1.0 = completo, 0.5 = mitad de canales)
            weights: Pesos pre-entrenados
            trainable: Si el backbone es entrenable
        """
        self.input_shape = input_shape
        self.alpha = alpha
        self.weights = weights
        self.trainable = trainable
        
        # Crear modelo base
        self.base_model = keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
            alpha=alpha
        )
        
        if not trainable:
            self.base_model.trainable = False
    
    def build(self) -> keras.Model:
        """
        Construir modelo de backbone.
        
        Returns:
            Modelo con salidas [C2, C3, C4, C5]
        """
        # Capas de salida para MobileNetV2
        output_layer_names = [
            'block_3_expand_relu',   # C2
            'block_6_expand_relu',   # C3
            'block_13_expand_relu',  # C4
            'out_relu',              # C5
        ]
        
        outputs = [
            self.base_model.get_layer(name).output
            for name in output_layer_names
        ]
        
        model = keras.Model(
            inputs=self.base_model.input,
            outputs=outputs,
            name='mobilenetv2_backbone'
        )
        
        return model
    
    def get_config(self) -> Dict:
        """Obtener configuración del backbone."""
        return {
            'type': 'mobilenetv2',
            'input_shape': self.input_shape,
            'alpha': self.alpha,
            'weights': self.weights,
            'trainable': self.trainable
        }
    
    def __repr__(self) -> str:
        """Representación en string."""
        return (f"MobileNetBackbone(input_shape={self.input_shape}, "
                f"alpha={self.alpha}, trainable={self.trainable})")
