"""
Red Neuronal Completamente Conectada para regresión de bounding box.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import List, Optional, Union
import numpy as np


class FCNetwork:
    """
    Red Neuronal Completamente Conectada para regresión de bounding box.
    
    Esta red toma vectores de características como entrada y genera coordenadas
    de bounding box normalizadas (x_center, y_center, width, height).
    """
    
    def __init__(
        self,
        input_dim: int,
        architecture: List[int] = [512, 256, 128, 64, 4],
        activations: Optional[List[str]] = None,
        use_batch_norm: bool = True,
        dropout_rates: Optional[List[float]] = None,
        l2_reg: float = 0.0,
        name: str = "FCNetwork"
    ):
        """
        Inicializar Red Completamente Conectada.
        
        Args:
            input_dim: Dimensión de características de entrada
            architecture: Lista de unidades por capa (la última debe ser 4 para bbox)
            activations: Función de activación para cada capa (por defecto: relu, ..., sigmoid)
            use_batch_norm: Si se debe usar normalización por lotes
            dropout_rates: Tasa de dropout para cada capa (None = sin dropout)
            l2_reg: Factor de regularización L2
            name: Nombre del modelo
        """
        self.input_dim = input_dim
        self.architecture = architecture
        self.use_batch_norm = use_batch_norm
        self.l2_reg = l2_reg
        self.name = name
        
        # Activaciones por defecto: ReLU para capas ocultas, Sigmoid para salida
        if activations is None:
            self.activations = ['relu'] * (len(architecture) - 1) + ['sigmoid']
        else:
            self.activations = activations
        
        # Tasas de dropout por defecto
        if dropout_rates is None:
            # Mayor dropout en capas tempranas, menor en capas posteriores
            self.dropout_rates = [0.3, 0.3, 0.2, 0.2] + [0.0] * (len(architecture) - 4)
        else:
            self.dropout_rates = dropout_rates
        
        # Asegurar que dropout_rates coincida con architecture
        while len(self.dropout_rates) < len(architecture):
            self.dropout_rates.append(0.0)
        
        # Construir modelo
        self.model = self._build_model()
    
    def _build_model(self) -> models.Model:
        """
        Construir la red completamente conectada.
        
        Returns:
            Modelo de Keras
        """
        # Capa de entrada
        inputs = layers.Input(shape=(self.input_dim,), name='input_features')
        x = inputs
        
        # Construir arquitectura
        for i, (units, activation, dropout) in enumerate(
            zip(self.architecture, self.activations, self.dropout_rates)
        ):
            layer_name = f'dense_{i+1}'
            
            # Capa densa con regularización L2 opcional
            if self.l2_reg > 0:
                x = layers.Dense(
                    units,
                    activation=None,  # Aplicar activación después de batch norm
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=layer_name
                )(x)
            else:
                x = layers.Dense(
                    units,
                    activation=None,
                    name=layer_name
                )(x)
            
            # Normalización por lotes (excepto para capa de salida)
            if self.use_batch_norm and i < len(self.architecture) - 1:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Activación
            x = layers.Activation(activation, name=f'activation_{i+1}')(x)
            
            # Dropout (excepto para capa de salida)
            if dropout > 0 and i < len(self.architecture) - 1:
                x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # Crear modelo
        model = models.Model(inputs=inputs, outputs=x, name=self.name)
        
        return model
    
    def compile(
        self,
        optimizer: Union[str, keras.optimizers.Optimizer] = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'mse',
        metrics: Optional[List[str]] = None
    ):
        """
        Compilar el modelo.
        
        Args:
            optimizer: Nombre o instancia del optimizador
            learning_rate: Tasa de aprendizaje
            loss: Función de pérdida
            metrics: Lista de métricas a rastrear
        """
        # Crear optimizador con tasa de aprendizaje
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                opt = keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'sgd':
                opt = keras.optimizers.SGD(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                opt = optimizer
        else:
            opt = optimizer
        
        # Métricas por defecto
        if metrics is None:
            metrics = ['mae']
        
        # Compilar modelo
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Imprimir resumen del modelo."""
        return self.model.summary()
    
    def get_model(self) -> models.Model:
        """Obtener el modelo de Keras."""
        return self.model
    
    def save(self, filepath: str):
        """Guardar modelo a archivo."""
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str) -> models.Model:
        """Cargar modelo desde archivo."""
        return keras.models.load_model(filepath, compile=False)
    
    def get_config(self) -> dict:
        """Obtener configuración del modelo."""
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'architecture': self.architecture,
            'activations': self.activations,
            'use_batch_norm': self.use_batch_norm,
            'dropout_rates': self.dropout_rates,
            'l2_reg': self.l2_reg,
            'total_params': self.model.count_params()
        }


def create_fc_model(
    input_dim: int,
    config: dict = None
) -> FCNetwork:
    """
    Función de fábrica para crear Red FC desde configuración.
    
    Args:
        input_dim: Dimensión de características de entrada
        config: Diccionario de configuración
        
    Returns:
        Instancia de FCNetwork
    """
    if config is None:
        # Configuración por defecto
        config = {
            'architecture': [512, 256, 128, 64, 4],
            'use_batch_norm': True,
            'dropout_rates': [0.3, 0.3, 0.2, 0.2, 0.0]
        }
    
    return FCNetwork(input_dim=input_dim, **config)
