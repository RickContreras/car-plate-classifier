"""
Feature Pyramid Network (FPN) para detección multi-escala.

FPN construye una pirámide de features con información semántica fuerte
en todos los niveles, combinando features de alta resolución (detalles)
con features de baja resolución (semántica).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict


class FeaturePyramidNetwork:
    """
    Feature Pyramid Network para detección a múltiples escalas.
    
    FPN combina features del backbone mediante:
    1. Lateral connections: proyecciones 1x1 de features del backbone
    2. Top-down pathway: upsampling de features de niveles superiores
    3. Skip connections: suma elemento-a-elemento
    4. Smoothing convolutions: conv 3x3 para reducir aliasing
    
    Input: [C2, C3, C4, C5] del backbone
    Output: [P3, P4, P5, P6, P7] con 256 canales cada uno
    
    Attributes:
        feature_size: Número de canales para todos los niveles (típicamente 256)
        num_levels: Número de niveles en la pirámide (5 para P3-P7)
    
    Referencias:
        Lin et al. "Feature Pyramid Networks for Object Detection" (2017)
        https://arxiv.org/abs/1612.03144
    """
    
    def __init__(
        self,
        feature_size: int = 256,
        num_levels: int = 5
    ):
        """
        Inicializar FPN.
        
        Args:
            feature_size: Número de canales de salida (típicamente 256)
            num_levels: Número de niveles P3, P4, P5, P6, P7 (típicamente 5)
        """
        self.feature_size = feature_size
        self.num_levels = num_levels
    
    def build(
        self,
        backbone_features: List[tf.Tensor]
    ) -> List[tf.Tensor]:
        """
        Construir Feature Pyramid a partir de features del backbone.
        
        Args:
            backbone_features: Lista [C2, C3, C4, C5] del backbone
            
        Returns:
            Lista [P3, P4, P5, P6, P7] de feature maps
        """
        C2, C3, C4, C5 = backbone_features
        
        # ===== Top-down pathway con lateral connections =====
        
        # P5: proyección 1x1 de C5
        P5 = layers.Conv2D(
            self.feature_size,
            kernel_size=1,
            strides=1,
            padding='same',
            name='fpn_c5p5'
        )(C5)
        
        # P4: lateral C4 + upsampled P5
        P5_upsampled = layers.UpSampling2D(
            size=(2, 2),
            interpolation='nearest',
            name='fpn_p5upsampled'
        )(P5)
        
        P4_lateral = layers.Conv2D(
            self.feature_size,
            kernel_size=1,
            strides=1,
            padding='same',
            name='fpn_c4p4'
        )(C4)
        
        P4 = layers.Add(name='fpn_p4add')([P5_upsampled, P4_lateral])
        
        # P3: lateral C3 + upsampled P4
        P4_upsampled = layers.UpSampling2D(
            size=(2, 2),
            interpolation='nearest',
            name='fpn_p4upsampled'
        )(P4)
        
        P3_lateral = layers.Conv2D(
            self.feature_size,
            kernel_size=1,
            strides=1,
            padding='same',
            name='fpn_c3p3'
        )(C3)
        
        P3 = layers.Add(name='fpn_p3add')([P4_upsampled, P3_lateral])
        
        # ===== Smoothing convolutions (reduce aliasing artifacts) =====
        
        P3 = layers.Conv2D(
            self.feature_size,
            kernel_size=3,
            strides=1,
            padding='same',
            name='fpn_p3'
        )(P3)
        
        P4 = layers.Conv2D(
            self.feature_size,
            kernel_size=3,
            strides=1,
            padding='same',
            name='fpn_p4'
        )(P4)
        
        P5 = layers.Conv2D(
            self.feature_size,
            kernel_size=3,
            strides=1,
            padding='same',
            name='fpn_p5'
        )(P5)
        
        # ===== Coarser levels P6, P7 para objetos grandes =====
        
        # P6: conv 3x3 stride 2 sobre C5
        P6 = layers.Conv2D(
            self.feature_size,
            kernel_size=3,
            strides=2,
            padding='same',
            name='fpn_p6'
        )(C5)
        
        # P7: ReLU + conv 3x3 stride 2 sobre P6
        P7 = layers.Activation('relu', name='fpn_p6relu')(P6)
        P7 = layers.Conv2D(
            self.feature_size,
            kernel_size=3,
            strides=2,
            padding='same',
            name='fpn_p7'
        )(P7)
        
        return [P3, P4, P5, P6, P7]
    
    def get_config(self) -> Dict:
        """
        Obtener configuración del FPN.
        
        Returns:
            Diccionario con parámetros
        """
        return {
            'feature_size': self.feature_size,
            'num_levels': self.num_levels
        }
    
    def __repr__(self) -> str:
        """Representación en string."""
        return (f"FeaturePyramidNetwork(feature_size={self.feature_size}, "
                f"num_levels={self.num_levels})")
