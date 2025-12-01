"""
Funciones de pérdida para RetinaNet.

Este módulo implementa las funciones de pérdida especializadas para
entrenamiento de RetinaNet, incluyendo Focal Loss para clasificación
y Smooth L1 Loss para regresión de bounding boxes.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Dict


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss para abordar class imbalance en detección de objetos.
    
    Focal Loss modifica Cross Entropy Loss para reducir el peso de ejemplos
    fáciles y enfocarse en casos difíciles, resolviendo el problema de
    desbalance entre clases (muchos backgrounds, pocos objetos).
    
    Fórmula: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Donde:
    - p_t: probabilidad predicha de la clase verdadera
    - α: factor de balance de clases (típicamente 0.25)
    - γ: factor de enfoque/focusing (típicamente 2.0)
    
    Attributes:
        alpha: Factor de balance para foreground vs background
        gamma: Parámetro de enfoque (γ=0 → Cross Entropy, γ>0 → Focal Loss)
        from_logits: Si las predicciones son logits o probabilidades
    
    Referencias:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        from_logits: bool = False,
        reduction: str = 'sum',
        name: str = 'focal_loss'
    ):
        """
        Inicializar Focal Loss.
        
        Args:
            alpha: Factor de balance (0.25 = más peso a positivos)
            gamma: Parámetro de enfoque (2.0 es valor estándar)
            from_logits: Si True, aplica sigmoid antes del cálculo
            reduction: Tipo de reducción ('sum', 'mean', 'none')
            name: Nombre de la pérdida
        """
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calcular Focal Loss.
        
        Args:
            y_true: Labels verdaderos (batch_size, num_anchors, num_classes)
            y_pred: Predicciones (batch_size, num_anchors, num_classes)
            
        Returns:
            Pérdida escalar o por muestra según reduction
        """
        # Convertir logits a probabilidades si es necesario
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip para estabilidad numérica
        epsilon = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcular cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calcular factor de modulación: (1 - p_t)^gamma
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        # Aplicar factor alpha para balance de clases
        alpha_factor = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        
        # Focal Loss = alpha * (1-p_t)^gamma * CE
        focal_loss = alpha_factor * modulating_factor * cross_entropy
        
        return tf.reduce_sum(focal_loss, axis=-1)
    
    def get_config(self) -> Dict:
        """
        Obtener configuración de la pérdida.
        
        Returns:
            Diccionario con parámetros de configuración
        """
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'from_logits': self.from_logits
        })
        return config


class SmoothL1Loss(keras.losses.Loss):
    """
    Smooth L1 Loss para regresión robusta de bounding boxes.
    
    Smooth L1 combina las ventajas de L1 y L2 loss:
    - Para errores pequeños (|x| < δ): usa L2 (cuadrático) → suave
    - Para errores grandes (|x| ≥ δ): usa L1 (lineal) → robusto a outliers
    
    Fórmula:
        smooth_L1(x) = {
            0.5 * x^2 / δ           si |x| < δ
            |x| - 0.5 * δ           en otro caso
        }
    
    Attributes:
        delta: Punto de transición entre L1 y L2
    
    Referencias:
        Girshick, "Fast R-CNN" (2015)
        https://arxiv.org/abs/1504.08083
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = 'sum',
        name: str = 'smooth_l1_loss'
    ):
        """
        Inicializar Smooth L1 Loss.
        
        Args:
            delta: Punto de transición (típicamente 1.0)
            reduction: Tipo de reducción
            name: Nombre de la pérdida
        """
        super().__init__(reduction=reduction, name=name)
        self.delta = delta
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calcular Smooth L1 Loss.
        
        Args:
            y_true: Coordenadas verdaderas (batch_size, num_anchors, 4)
            y_pred: Coordenadas predichas (batch_size, num_anchors, 4)
            
        Returns:
            Pérdida escalar o por muestra
        """
        diff = y_pred - y_true
        abs_diff = tf.abs(diff)
        
        # Aplicar función Smooth L1
        loss = tf.where(
            abs_diff < self.delta,
            0.5 * tf.square(diff) / self.delta,
            abs_diff - 0.5 * self.delta
        )
        
        return tf.reduce_sum(loss, axis=-1)
    
    def get_config(self) -> Dict:
        """
        Obtener configuración de la pérdida.
        
        Returns:
            Diccionario con parámetros
        """
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


class RetinaNetLoss(keras.losses.Loss):
    """
    Pérdida combinada para entrenamiento de RetinaNet.
    
    Combina Focal Loss para clasificación y Smooth L1 Loss para regresión:
    
    Total Loss = L_cls + λ * L_box
    
    Donde:
    - L_cls: Focal Loss para todas las anchors
    - L_box: Smooth L1 Loss solo para anchors positivas (con objeto)
    - λ: factor de balance (típicamente 1.0)
    
    Attributes:
        num_classes: Número de clases a detectar
        lambda_box: Factor de ponderación para pérdida de regresión
        focal_loss: Instancia de FocalLoss
        smooth_l1_loss: Instancia de SmoothL1Loss
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 1.0,
        lambda_box: float = 1.0,
        name: str = 'retinanet_loss'
    ):
        """
        Inicializar pérdida combinada de RetinaNet.
        
        Args:
            num_classes: Número de clases (1 para placas vehiculares)
            alpha: Factor alpha para Focal Loss
            gamma: Factor gamma para Focal Loss
            delta: Delta para Smooth L1 Loss
            lambda_box: Factor de balance para pérdida de regresión
            name: Nombre de la pérdida
        """
        super().__init__(name=name)
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        
        # Inicializar pérdidas individuales
        self.focal_loss = FocalLoss(
            alpha=alpha,
            gamma=gamma,
            from_logits=True,
            reduction='none'
        )
        self.smooth_l1_loss = SmoothL1Loss(
            delta=delta,
            reduction='none'
        )
    
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Calcular pérdida total de RetinaNet.
        
        Args:
            y_true: Tupla de (class_targets, box_targets, positive_mask)
                - class_targets: (batch, num_anchors, num_classes)
                - box_targets: (batch, num_anchors, 4)
                - positive_mask: (batch, num_anchors) máscara booleana
            y_pred: Tupla de (class_predictions, box_predictions)
                - class_predictions: (batch, num_anchors, num_classes)
                - box_predictions: (batch, num_anchors, 4)
                
        Returns:
            Pérdida total promedio sobre el batch
        """
        # Desempaquetar targets y predicciones
        class_targets, box_targets, positive_mask = y_true
        class_preds, box_preds = y_pred
        
        # Convertir positive_mask a float32 si es necesario
        positive_mask = tf.cast(positive_mask, tf.float32)
        
        # 1. Pérdida de clasificación (para todas las anchors)
        cls_loss = self.focal_loss(class_targets, class_preds)
        
        # 2. Pérdida de regresión (solo para positive anchors)
        # Expandir máscara para cada coordenada (4 valores)
        positive_mask_expanded = tf.expand_dims(positive_mask, axis=-1)
        positive_mask_expanded = tf.tile(positive_mask_expanded, [1, 1, 4])
        
        # Aplicar máscara a targets y predicciones
        box_targets_masked = box_targets * positive_mask_expanded
        box_preds_masked = box_preds * positive_mask_expanded
        
        # Calcular pérdida de regresión
        box_loss = self.smooth_l1_loss(box_targets_masked, box_preds_masked)
        
        # 3. Normalizar por número de positive anchors
        # Evitar división por cero
        num_positives = tf.reduce_sum(positive_mask, axis=1)
        num_positives = tf.maximum(num_positives, 1.0)
        
        # Normalizar pérdidas
        cls_loss = tf.reduce_sum(cls_loss, axis=1) / num_positives
        box_loss = tf.reduce_sum(box_loss, axis=1) / num_positives
        
        # 4. Pérdida total combinada
        total_loss = cls_loss + self.lambda_box * box_loss
        
        # Promedio sobre el batch
        return tf.reduce_mean(total_loss)
    
    def get_config(self) -> Dict:
        """
        Obtener configuración de la pérdida.
        
        Returns:
            Diccionario con todos los parámetros
        """
        return {
            'num_classes': self.num_classes,
            'lambda_box': self.lambda_box,
            'alpha': self.focal_loss.alpha,
            'gamma': self.focal_loss.gamma,
            'delta': self.smooth_l1_loss.delta
        }
    
    def __repr__(self) -> str:
        """Representación en string de la pérdida."""
        return (f"RetinaNetLoss(num_classes={self.num_classes}, "
                f"alpha={self.focal_loss.alpha}, "
                f"gamma={self.focal_loss.gamma}, "
                f"lambda_box={self.lambda_box})")
