"""
Fully Connected Neural Network for bounding box regression.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from typing import List, Optional, Union
import numpy as np


class FCNetwork:
    """
    Fully Connected Neural Network for bounding box regression.
    
    This network takes feature vectors as input and outputs normalized
    bounding box coordinates (x_center, y_center, width, height).
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
        Initialize Fully Connected Network.
        
        Args:
            input_dim: Dimension of input features
            architecture: List of units per layer (last should be 4 for bbox)
            activations: Activation function for each layer (default: relu, ..., sigmoid)
            use_batch_norm: Whether to use batch normalization
            dropout_rates: Dropout rate for each layer (None = no dropout)
            l2_reg: L2 regularization factor
            name: Name of the model
        """
        self.input_dim = input_dim
        self.architecture = architecture
        self.use_batch_norm = use_batch_norm
        self.l2_reg = l2_reg
        self.name = name
        
        # Default activations: ReLU for hidden layers, Sigmoid for output
        if activations is None:
            self.activations = ['relu'] * (len(architecture) - 1) + ['sigmoid']
        else:
            self.activations = activations
        
        # Default dropout rates
        if dropout_rates is None:
            # Higher dropout in early layers, lower in later layers
            self.dropout_rates = [0.3, 0.3, 0.2, 0.2] + [0.0] * (len(architecture) - 4)
        else:
            self.dropout_rates = dropout_rates
        
        # Ensure dropout_rates matches architecture
        while len(self.dropout_rates) < len(architecture):
            self.dropout_rates.append(0.0)
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> models.Model:
        """
        Build the fully connected network.
        
        Returns:
            Keras Model
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input_features')
        x = inputs
        
        # Build architecture
        for i, (units, activation, dropout) in enumerate(
            zip(self.architecture, self.activations, self.dropout_rates)
        ):
            layer_name = f'dense_{i+1}'
            
            # Dense layer with optional L2 regularization
            if self.l2_reg > 0:
                x = layers.Dense(
                    units,
                    activation=None,  # Apply activation after batch norm
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=layer_name
                )(x)
            else:
                x = layers.Dense(
                    units,
                    activation=None,
                    name=layer_name
                )(x)
            
            # Batch normalization (except for output layer)
            if self.use_batch_norm and i < len(self.architecture) - 1:
                x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            
            # Activation
            x = layers.Activation(activation, name=f'activation_{i+1}')(x)
            
            # Dropout (except for output layer)
            if dropout > 0 and i < len(self.architecture) - 1:
                x = layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # Create model
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
        Compile the model.
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics to track
        """
        # Create optimizer with learning rate
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
        
        # Default metrics
        if metrics is None:
            metrics = ['mae']
        
        # Compile model
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
    
    def summary(self):
        """Print model summary."""
        return self.model.summary()
    
    def get_model(self) -> models.Model:
        """Get the Keras model."""
        return self.model
    
    def save(self, filepath: str):
        """Save model to file."""
        self.model.save(filepath)
    
    @staticmethod
    def load(filepath: str) -> models.Model:
        """Load model from file."""
        return keras.models.load_model(filepath, compile=False)
    
    def get_config(self) -> dict:
        """Get model configuration."""
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
    Factory function to create FC Network from configuration.
    
    Args:
        input_dim: Input feature dimension
        config: Configuration dictionary
        
    Returns:
        FCNetwork instance
    """
    if config is None:
        # Default configuration
        config = {
            'architecture': [512, 256, 128, 64, 4],
            'use_batch_norm': True,
            'dropout_rates': [0.3, 0.3, 0.2, 0.2, 0.0]
        }
    
    return FCNetwork(input_dim=input_dim, **config)
