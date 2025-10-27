"""
Clase entrenadora para entrenar modelos de detección.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime


class Trainer:
    """Clase entrenadora para modelos de detección."""
    
    def __init__(
        self,
        model: keras.Model,
        save_dir: str = "models",
        log_dir: str = "logs",
        name: str = "detection_model"
    ):
        """
        Inicializar entrenador.
        
        Args:
            model: Modelo de Keras a entrenar
            save_dir: Directorio para guardar modelos
            log_dir: Directorio para logs
            name: Nombre del modelo
        """
        self.model = model
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)
        self.name = name
        
        # Crear directorios
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
        Entrenar el modelo.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento (bboxes)
            X_val: Características de validación
            y_val: Etiquetas de validación
            epochs: Número de épocas
            batch_size: Tamaño de lote
            callbacks: Lista de callbacks
            verbose: Modo de verbosidad
            
        Returns:
            Historial de entrenamiento
        """
        print(f"\n{'='*60}")
        print(f"Entrenando {self.name}")
        print(f"{'='*60}")
        print(f"Muestras de entrenamiento: {len(X_train)}")
        print(f"Muestras de validación: {len(X_val)}")
        print(f"Épocas: {epochs}")
        print(f"Tamaño de lote: {batch_size}")
        print(f"{'='*60}\n")
        
        # Entrenar modelo
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
        Evaluar el modelo.
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            batch_size: Tamaño de lote
            
        Returns:
            Diccionario de métricas
        """
        print(f"\n{'='*60}")
        print(f"Evaluando {self.name}")
        print(f"{'='*60}")
        
        results = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
        
        # Crear diccionario de métricas
        metrics = {}
        for name, value in zip(self.model.metrics_names, results):
            metrics[name] = float(value)
        
        print(f"\nResultados de Prueba:")
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
        Realizar predicciones.
        
        Args:
            X: Características
            batch_size: Tamaño de lote
            
        Returns:
            Predicciones
        """
        return self.model.predict(X, batch_size=batch_size, verbose=0)
    
    def save_model(self, filepath: Optional[str] = None):
        """
        Guardar modelo a archivo.
        
        Args:
            filepath: Ruta para guardar modelo (por defecto: save_dir/name.h5)
        """
        if filepath is None:
            filepath = self.save_dir / f"{self.name}.h5"
        
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def save_history(self, filepath: Optional[str] = None):
        """
        Guardar historial de entrenamiento a JSON.
        
        Args:
            filepath: Ruta para guardar historial (por defecto: save_dir/name_history.json)
        """
        if self.history is None:
            print("No hay historial de entrenamiento para guardar")
            return
        
        if filepath is None:
            filepath = self.save_dir / f"{self.name}_history.json"
        
        # Convertir historial a diccionario
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        # Guardar a JSON
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"Historial guardado en: {filepath}")
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """
        Obtener la época con la mejor métrica de validación.
        
        Args:
            metric: Métrica a usar
            mode: 'min' o 'max'
            
        Returns:
            Número de la mejor época
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
        """Imprimir resumen del modelo."""
        self.model.summary()
