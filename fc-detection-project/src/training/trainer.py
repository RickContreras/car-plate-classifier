"""
Trainer class for training detection models.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime


class Trainer:
    """Trainer class for detection models."""
    
    def __init__(
        self,
        model: keras.Model,
        save_dir: str = "models",
        log_dir: str = "logs",
        name: str = "detection_model"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            save_dir: Directory to save models
            log_dir: Directory for logs
            name: Model name
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.name = name
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: Optional[List] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels (bboxes)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks: List of callbacks
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name}")
        print(f"{'='*60}")
        print(f"Train samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate the model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            batch_size: Batch size
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.name}")
        print(f"{'='*60}")
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        
        # Create metrics dict
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        print(f"\nTest Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.6f}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            batch_size: Batch size
            
        Returns:
            Predictions
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model (default: save_dir/name.h5)
        """
        if filepath is None:
            filepath = self.save_dir / f"{self.name}.h5"
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def save_history(self, filepath: Optional[str] = None):
        """
        Save training history to JSON.
        
        Args:
            filepath: Path to save history (default: save_dir/name_history.json)
        """
        if self.history is None:
            print("No training history to save")
            return
        
        if filepath is None:
            filepath = self.save_dir / f"{self.name}_history.json"
        
        # Convert history to dict
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"History saved to: {filepath}")
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """
        Get the epoch with best validation metric.
        
        Args:
            metric: Metric to use
            mode: 'min' or 'max'
            
        Returns:
            Best epoch number
        """
        if self.history is None:
            return -1
        
        values = self.history.history[metric]
        
        if mode == 'min':
            best_epoch = np.argmin(values)
        else:
            best_epoch = np.argmax(values)
        
        return int(best_epoch)
    
    def summary(self):
        """Print model summary."""
        self.model.summary()
