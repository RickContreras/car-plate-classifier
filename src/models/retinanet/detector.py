"""
Detector RetinaNet completo para detección de objetos.

Este módulo ensambla todos los componentes (backbone, FPN, classification head,
box regression head) en un modelo end-to-end para detección de objetos.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Dict, Optional

from .backbone import ResNetBackbone, MobileNetBackbone
from .fpn import FeaturePyramidNetwork
from .anchors import AnchorGenerator
from .losses import RetinaNetLoss


class RetinaNetDetector:
    """
    Detector RetinaNet completo para detección de objetos.
    
    RetinaNet es un detector one-stage que utiliza:
    1. Backbone (ResNet/MobileNet) para extracción de features
    2. FPN para features multi-escala
    3. Classification subnet para predecir presencia de objetos
    4. Box regression subnet para predecir coordenadas
    5. Focal Loss para abordar class imbalance
    
    Arquitectura similar a FCNetwork pero end-to-end con features aprendidas.
    
    Attributes:
        num_classes: Número de clases a detectar
        input_shape: Forma de entrada (height, width, channels)
        backbone_type: Tipo de backbone ('resnet50', 'mobilenetv2')
        feature_size: Número de canales en FPN
        num_anchors: Número de anchors por ubicación
    
    Referencias:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        backbone_type: str = 'resnet50',
        backbone_weights: str = 'imagenet',
        backbone_trainable: bool = False,
        feature_size: int = 256,
        num_conv_layers: int = 4,
        anchor_sizes: List[float] = [32, 64, 128, 256, 512],
        anchor_scales: List[float] = [1.0, 1.26, 1.59],
        anchor_ratios: List[float] = [0.5, 1.0, 2.0],
        name: str = 'retinanet'
    ):
        """
        Inicializar detector RetinaNet.
        
        Args:
            num_classes: Número de clases (1 para placas vehiculares)
            input_shape: Forma de entrada (H, W, C)
            backbone_type: 'resnet50' o 'mobilenetv2'
            backbone_weights: 'imagenet', None, o ruta a pesos
            backbone_trainable: Si True, fine-tune del backbone
            feature_size: Canales en FPN (típicamente 256)
            num_conv_layers: Capas en classification/box subnets (típicamente 4)
            anchor_sizes: Tamaños base de anchors para P3-P7
            anchor_scales: Escalas por anchor
            anchor_ratios: Proporciones de aspecto
            name: Nombre del modelo
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.backbone_type = backbone_type
        self.backbone_weights = backbone_weights
        self.backbone_trainable = backbone_trainable
        self.feature_size = feature_size
        self.num_conv_layers = num_conv_layers
        self.name = name
        
        # Configurar generador de anchors
        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            scales=anchor_scales,
            aspect_ratios=anchor_ratios
        )
        self.num_anchors = self.anchor_generator.num_anchors
        
        # Inicializar componentes
        self._build_backbone()
        self.fpn = FeaturePyramidNetwork(feature_size=feature_size)
    
    def _build_backbone(self):
        """Construir backbone según tipo especificado."""
        if self.backbone_type == 'resnet50':
            self.backbone = ResNetBackbone(
                input_shape=self.input_shape,
                weights=self.backbone_weights,
                trainable=self.backbone_trainable
            )
        elif self.backbone_type == 'mobilenetv2':
            self.backbone = MobileNetBackbone(
                input_shape=self.input_shape,
                weights=self.backbone_weights,
                trainable=self.backbone_trainable
            )
        else:
            raise ValueError(f"Backbone no soportado: {self.backbone_type}")
    
    def _build_classification_subnet(
        self,
        num_layers: int = 4,
        num_filters: int = 256,
        name: str = 'classification_subnet'
    ) -> keras.Model:
        """
        Construir subnet de clasificación.
        
        Subnet compartido para predecir probabilidades de clase en todos
        los niveles de pirámide.
        
        Args:
            num_layers: Número de capas convolucionales
            num_filters: Filtros por capa
            name: Nombre del subnet
            
        Returns:
            Modelo de Keras para clasificación
        """
        inputs = layers.Input(shape=(None, None, self.feature_size))
        x = inputs
        
        # Stack de convoluciones con ReLU
        for i in range(num_layers):
            x = layers.Conv2D(
                num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                name=f'{name}_conv{i+1}'
            )(x)
        
        # Capa de salida: num_anchors * num_classes
        # Inicialización especial para mejorar convergencia inicial
        prior_prob = 0.01
        bias_init = -tf.math.log((1 - prior_prob) / prior_prob)
        
        outputs = layers.Conv2D(
            self.num_anchors * self.num_classes,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=keras.initializers.Constant(bias_init),
            name=f'{name}_output'
        )(x)
        
        # Reshape: (batch, H, W, num_anchors * num_classes) -> (batch, H*W*num_anchors, num_classes)
        outputs = layers.Reshape((-1, self.num_classes), name=f'{name}_reshape')(outputs)
        
        return keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    def _build_box_subnet(
        self,
        num_layers: int = 4,
        num_filters: int = 256,
        name: str = 'box_subnet'
    ) -> keras.Model:
        """
        Construir subnet de regresión de bounding boxes.
        
        Subnet compartido para predecir offsets de bounding boxes en todos
        los niveles de pirámide.
        
        Args:
            num_layers: Número de capas convolucionales
            num_filters: Filtros por capa
            name: Nombre del subnet
            
        Returns:
            Modelo de Keras para regresión
        """
        inputs = layers.Input(shape=(None, None, self.feature_size))
        x = inputs
        
        # Stack de convoluciones con ReLU
        for i in range(num_layers):
            x = layers.Conv2D(
                num_filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu',
                kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                name=f'{name}_conv{i+1}'
            )(x)
        
        # Capa de salida: num_anchors * 4 (dx, dy, dw, dh)
        outputs = layers.Conv2D(
            self.num_anchors * 4,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            name=f'{name}_output'
        )(x)
        
        # Reshape: (batch, H, W, num_anchors * 4) -> (batch, H*W*num_anchors, 4)
        outputs = layers.Reshape((-1, 4), name=f'{name}_reshape')(outputs)
        
        return keras.Model(inputs=inputs, outputs=outputs, name=name)
    
    def build(self) -> keras.Model:
        """
        Construir modelo RetinaNet completo.
        
        Returns:
            Modelo de Keras compilado y listo para entrenar
        """
        # 1. Input
        inputs = layers.Input(shape=self.input_shape, name='image_input')
        
        # 2. Backbone: extraer features a múltiples escalas
        backbone_model = self.backbone.build()
        C2, C3, C4, C5 = backbone_model(inputs)
        
        # 3. FPN: construir pirámide de features
        pyramid_features = self.fpn.build([C2, C3, C4, C5])  # [P3, P4, P5, P6, P7]
        
        # 4. Construir subnets de clasificación y regresión
        cls_subnet = self._build_classification_subnet(self.num_conv_layers, self.feature_size)
        box_subnet = self._build_box_subnet(self.num_conv_layers, self.feature_size)
        
        # 5. Aplicar subnets a cada nivel de pirámide
        cls_outputs = []
        box_outputs = []
        
        for level_idx, P in enumerate(pyramid_features):
            cls_out = cls_subnet(P)
            box_out = box_subnet(P)
            
            cls_outputs.append(cls_out)
            box_outputs.append(box_out)
        
        # 6. Concatenar predicciones de todos los niveles
        cls_predictions = layers.Concatenate(axis=1, name='classification_concat')(cls_outputs)
        box_predictions = layers.Concatenate(axis=1, name='box_concat')(box_outputs)
        
        # 7. Crear modelo final
        model = keras.Model(
            inputs=inputs,
            outputs=[cls_predictions, box_predictions],
            name=self.name
        )
        
        return model
    
    def build_for_inference(self) -> keras.Model:
        """
        Construir modelo RetinaNet solo para inferencia (sin el wrapper de entrenamiento).
        
        Returns:
            Modelo base de Keras listo para inferencia
        """
        return self.build()
    
    def compile_model(
        self,
        model: keras.Model,
        learning_rate: float = 1e-4,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_box: float = 1.0
    ) -> keras.Model:
        """
        Compilar modelo con optimizador y función de pérdida.
        
        Args:
            model: Modelo a compilar
            learning_rate: Tasa de aprendizaje
            alpha: Factor alpha para Focal Loss
            gamma: Factor gamma para Focal Loss
            lambda_box: Factor de balance para box loss
            
        Returns:
            Modelo compilado (envuelto en RetinaNetModel)
        """
        from .losses import RetinaNetLoss
        
        # Crear modelo personalizado con training step
        class RetinaNetModel(keras.Model):
            def __init__(self, base_model, loss_fn):
                super().__init__()
                self.base_model = base_model
                self.loss_fn = loss_fn
                
                # Crear métrica para trackear la pérdida
                self.loss_tracker = keras.metrics.Mean(name="loss")
            
            @property
            def metrics(self):
                # Retornar las métricas que queremos trackear
                return [self.loss_tracker]
            
            def call(self, inputs, training=False):
                return self.base_model(inputs, training=training)
            
            def train_step(self, data):
                x, y_true = data
                
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = self.base_model(x, training=True)
                    # Calcular pérdida
                    loss = self.loss_fn(y_true, y_pred)
                
                # Calcular gradientes
                trainable_vars = self.base_model.trainable_variables
                gradients = tape.gradient(loss, trainable_vars)
                
                # Actualizar pesos
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
                # Actualizar métrica de pérdida
                self.loss_tracker.update_state(loss)
                
                # Retornar todas las métricas
                return {"loss": self.loss_tracker.result()}
            
            def test_step(self, data):
                x, y_true = data
                
                # Forward pass
                y_pred = self.base_model(x, training=False)
                
                # Calcular pérdida
                loss = self.loss_fn(y_true, y_pred)
                
                # Actualizar métrica de pérdida
                self.loss_tracker.update_state(loss)
                
                # Retornar todas las métricas
                return {"loss": self.loss_tracker.result()}
        
        # Crear función de pérdida
        loss_fn = RetinaNetLoss(
            num_classes=self.num_classes,
            alpha=alpha,
            gamma=gamma,
            lambda_box=lambda_box
        )
        
        # Crear modelo personalizado
        custom_model = RetinaNetModel(model, loss_fn)
        
        # Compilar con optimizador
        custom_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
        )
        
        return custom_model
    
    def get_config(self) -> Dict:
        """
        Obtener configuración del detector.
        
        Returns:
            Diccionario con toda la configuración
        """
        return {
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'backbone_type': self.backbone_type,
            'backbone_weights': self.backbone_weights,
            'backbone_trainable': self.backbone_trainable,
            'feature_size': self.feature_size,
            'num_conv_layers': self.num_conv_layers,
            'num_anchors': self.num_anchors,
            'anchor_config': self.anchor_generator.get_config(),
            'name': self.name
        }
    
    def summary(self):
        """Mostrar resumen de la arquitectura."""
        model = self.build()
        model.summary()
        
        print("\n" + "="*70)
        print("CONFIGURACIÓN RETINANET")
        print("="*70)
        config = self.get_config()
        for key, value in config.items():
            if key != 'anchor_config':
                print(f"{key:.<30} {value}")
        print("="*70)
    
    def __repr__(self) -> str:
        """Representación en string del detector."""
        return (f"RetinaNetDetector(num_classes={self.num_classes}, "
                f"backbone='{self.backbone_type}', "
                f"input_shape={self.input_shape}, "
                f"num_anchors={self.num_anchors})")
