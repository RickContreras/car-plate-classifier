"""
Generador de Anchor Boxes para RetinaNet.

Este módulo implementa la generación de cajas de anclaje (anchor boxes)
que sirven como referencias para predecir bounding boxes finales.
"""

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict


class AnchorGenerator:
    """
    Generador de anchor boxes para detección multi-escala.
    
    Las anchor boxes son cajas predefinidas de diferentes tamaños y proporciones
    que se colocan en cada ubicación del feature map. El modelo predice ajustes
    (deltas) sobre estas anchors para obtener las predicciones finales.
    
    Attributes:
        sizes: Tamaño base para cada nivel de pirámide (P3-P7)
        scales: Escalas adicionales por anchor (típicamente 2^(0/3), 2^(1/3), 2^(2/3))
        aspect_ratios: Proporciones de aspecto (ancho/alto)
        strides: Stride de cada nivel de pirámide
        num_anchors: Número total de anchors por posición
    """
    
    def __init__(
        self,
        sizes: List[float] = [32, 64, 128, 256, 512],
        scales: List[float] = [1.0, 1.26, 1.59],
        aspect_ratios: List[float] = [0.5, 1.0, 2.0],
        strides: List[int] = [8, 16, 32, 64, 128]
    ):
        """
        Inicializar generador de anchors.
        
        Args:
            sizes: Tamaño base para cada nivel P3, P4, P5, P6, P7
            scales: Escalas multiplicativas para cada anchor
            aspect_ratios: Proporciones ancho/alto (0.5 = vertical, 2.0 = horizontal)
            strides: Stride (downsampling factor) de cada nivel
        """
        self.sizes = sizes
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.strides = strides
        self.num_anchors = len(scales) * len(aspect_ratios)
        
        # Validar configuración
        if len(sizes) != len(strides):
            raise ValueError(f"sizes y strides deben tener la misma longitud")
    
    def generate_anchors(
        self,
        image_shape: Tuple[int, int]
    ) -> tf.Tensor:
        """
        Generar todas las anchor boxes para una imagen.
        
        Args:
            image_shape: Tupla (height, width) de la imagen de entrada
            
        Returns:
            Tensor de forma (num_anchors_total, 4) con formato (x1, y1, x2, y2)
        """
        height, width = image_shape
        all_anchors = []
        
        # Generar anchors para cada nivel de pirámide
        for size, stride in zip(self.sizes, self.strides):
            # Calcular tamaño del feature map en este nivel
            fm_height = height // stride
            fm_width = width // stride
            
            # Generar anchors para este nivel
            level_anchors = self._generate_level_anchors(
                size, stride, fm_height, fm_width
            )
            all_anchors.append(level_anchors)
        
        # Concatenar anchors de todos los niveles
        anchors = tf.concat(all_anchors, axis=0)
        
        return anchors
    
    def _generate_level_anchors(
        self,
        size: float,
        stride: int,
        fm_height: int,
        fm_width: int
    ) -> tf.Tensor:
        """
        Generar anchors para un nivel específico de la pirámide.
        
        Args:
            size: Tamaño base del anchor para este nivel
            stride: Stride del feature map
            fm_height: Altura del feature map
            fm_width: Ancho del feature map
            
        Returns:
            Tensor de anchors (fm_height * fm_width * num_anchors, 4)
        """
        # Crear grid de centros de anchors
        cx = tf.range(fm_width, dtype=tf.float32) * stride + stride / 2.0
        cy = tf.range(fm_height, dtype=tf.float32) * stride + stride / 2.0
        cx_grid, cy_grid = tf.meshgrid(cx, cy)
        
        # Aplanar grids
        centers = tf.stack([
            tf.reshape(cx_grid, [-1]),
            tf.reshape(cy_grid, [-1])
        ], axis=1)  # Shape: (fm_height * fm_width, 2)
        
        # Generar anchors base para todas las escalas y proporciones
        base_anchors = self._generate_base_anchors(size)  # Shape: (num_anchors, 4)
        
        # Expandir centros y anchors para combinarlos
        num_centers = tf.shape(centers)[0]
        
        # Repetir centros para cada anchor
        centers_expanded = tf.expand_dims(centers, axis=1)  # (num_centers, 1, 2)
        centers_expanded = tf.tile(centers_expanded, [1, self.num_anchors, 1])
        centers_expanded = tf.reshape(centers_expanded, [-1, 2])
        
        # Repetir anchors base para cada centro
        base_anchors_expanded = tf.expand_dims(base_anchors, axis=0)  # (1, num_anchors, 4)
        base_anchors_expanded = tf.tile(base_anchors_expanded, [num_centers, 1, 1])
        base_anchors_expanded = tf.reshape(base_anchors_expanded, [-1, 4])
        
        # Convertir de formato (cx_offset, cy_offset, w, h) a (x1, y1, x2, y2)
        x1 = centers_expanded[:, 0] + base_anchors_expanded[:, 0] - base_anchors_expanded[:, 2] / 2
        y1 = centers_expanded[:, 1] + base_anchors_expanded[:, 1] - base_anchors_expanded[:, 3] / 2
        x2 = centers_expanded[:, 0] + base_anchors_expanded[:, 0] + base_anchors_expanded[:, 2] / 2
        y2 = centers_expanded[:, 1] + base_anchors_expanded[:, 1] + base_anchors_expanded[:, 3] / 2
        
        anchors = tf.stack([x1, y1, x2, y2], axis=1)
        
        return anchors
    
    def _generate_base_anchors(self, size: float) -> tf.Tensor:
        """
        Generar anchors base para todas las escalas y proporciones.
        
        Los anchors base están centrados en (0, 0) y se definen por
        su ancho y alto según las escalas y proporciones configuradas.
        
        Args:
            size: Tamaño base del anchor
            
        Returns:
            Tensor de forma (num_anchors, 4) con formato (cx_offset, cy_offset, w, h)
        """
        anchors = []
        
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                # Calcular ancho y alto según la proporción
                # ratio = w/h, por lo tanto: w = size * scale * sqrt(ratio)
                #                             h = size * scale / sqrt(ratio)
                w = size * scale * np.sqrt(ratio)
                h = size * scale / np.sqrt(ratio)
                
                # Anchor centrado en (0, 0) relativo al centro de la celda
                anchors.append([0.0, 0.0, w, h])
        
        return tf.constant(anchors, dtype=tf.float32)
    
    def get_config(self) -> Dict:
        """
        Obtener configuración del generador de anchors.
        
        Returns:
            Diccionario con la configuración
        """
        return {
            'sizes': self.sizes,
            'scales': self.scales,
            'aspect_ratios': self.aspect_ratios,
            'strides': self.strides,
            'num_anchors': self.num_anchors
        }
    
    def __repr__(self) -> str:
        """Representación en string del generador."""
        return (f"AnchorGenerator(sizes={self.sizes}, "
                f"scales={self.scales}, "
                f"aspect_ratios={self.aspect_ratios}, "
                f"num_anchors={self.num_anchors})")


def encode_boxes(
    gt_boxes: tf.Tensor,
    anchors: tf.Tensor,
    variances: List[float] = [0.1, 0.1, 0.2, 0.2]
) -> tf.Tensor:
    """
    Codificar ground truth boxes como offsets relativos a anchors.
    
    Esta codificación facilita el aprendizaje al convertir coordenadas absolutas
    en offsets normalizados respecto a las anchors.
    
    Args:
        gt_boxes: Ground truth boxes (N, 4) en formato (x1, y1, x2, y2)
        anchors: Anchor boxes (N, 4) en formato (x1, y1, x2, y2)
        variances: Factores de escala para normalizar los offsets
        
    Returns:
        Deltas codificados (N, 4) en formato (dx, dy, dw, dh)
    """
    # Convertir formato (x1, y1, x2, y2) a (cx, cy, w, h)
    gt_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    gt_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Evitar división por cero
    anchor_w = tf.maximum(anchor_w, 1e-6)
    anchor_h = tf.maximum(anchor_h, 1e-6)
    
    # Calcular deltas normalizados
    # dx, dy: offset del centro normalizado por el tamaño del anchor
    # dw, dh: escala logarítmica del tamaño
    dx = (gt_cx - anchor_cx) / anchor_w / variances[0]
    dy = (gt_cy - anchor_cy) / anchor_h / variances[1]
    dw = tf.math.log(gt_w / anchor_w) / variances[2]
    dh = tf.math.log(gt_h / anchor_h) / variances[3]
    
    return tf.stack([dx, dy, dw, dh], axis=1)


def decode_boxes(
    deltas: tf.Tensor,
    anchors: tf.Tensor,
    variances: List[float] = [0.1, 0.1, 0.2, 0.2]
) -> tf.Tensor:
    """
    Decodificar deltas predichos a bounding boxes absolutas.
    
    Convierte los offsets predichos por el modelo de vuelta a coordenadas
    absolutas en la imagen.
    
    Args:
        deltas: Deltas predichos (N, 4) en formato (dx, dy, dw, dh)
        anchors: Anchor boxes (N, 4) en formato (x1, y1, x2, y2)
        variances: Factores de escala usados en la codificación
        
    Returns:
        Bounding boxes (N, 4) en formato (x1, y1, x2, y2)
    """
    # Convertir anchors a formato (cx, cy, w, h)
    anchor_cx = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_cy = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    
    # Evitar división por cero
    anchor_w = tf.maximum(anchor_w, 1e-6)
    anchor_h = tf.maximum(anchor_h, 1e-6)
    
    # Decodificar deltas
    dx = deltas[:, 0] * variances[0]
    dy = deltas[:, 1] * variances[1]
    dw = deltas[:, 2] * variances[2]
    dh = deltas[:, 3] * variances[3]
    
    # Calcular coordenadas predichas
    pred_cx = dx * anchor_w + anchor_cx
    pred_cy = dy * anchor_h + anchor_cy
    pred_w = tf.exp(dw) * anchor_w
    pred_h = tf.exp(dh) * anchor_h
    
    # Convertir de vuelta a formato (x1, y1, x2, y2)
    x1 = pred_cx - pred_w / 2.0
    y1 = pred_cy - pred_h / 2.0
    x2 = pred_cx + pred_w / 2.0
    y2 = pred_cy + pred_h / 2.0
    
    return tf.stack([x1, y1, x2, y2], axis=1)
