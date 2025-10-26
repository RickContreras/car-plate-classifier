"""
Módulo de entrenamiento de modelos para clasificación de placas vehiculares.
Incluye SVM, Random Forest y Red Neuronal.
"""

import os
import numpy as np
import yaml
import joblib
from pathlib import Path
from typing import Dict, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga la configuración desde el archivo YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_neural_network(input_shape: int, config: dict = None) -> keras.Model:
    """
    Crea una red neuronal para clasificación de placas.
    
    Args:
        input_shape: Tamaño del vector de entrada
        config: Configuración del modelo
        
    Returns:
        keras.Model: Modelo de red neuronal compilado
    """
    if config and 'model' in config and 'neural_network' in config['model']:
        nn_config = config['model']['neural_network']
        hidden_layers = nn_config.get('hidden_layers', [256, 128, 64])
        dropout_rates = nn_config.get('dropout_rates', [0.5, 0.3, 0.2])
        learning_rate = nn_config.get('learning_rate', 0.001)
    else:
        hidden_layers = [256, 128, 64]
        dropout_rates = [0.5, 0.3, 0.2]
        learning_rate = 0.001
    
    # Construir modelo dinámicamente
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    
    for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))
    
    model.add(layers.Dense(1, activation='sigmoid', name='output'))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class PlateClassifier:
    """Clasificador de placas vehiculares."""
    
    def __init__(self, model_type: str = 'svm', config: dict = None):
        """
        Inicializa el clasificador.
        
        Args:
            model_type: Tipo de modelo ('svm', 'random_forest', 'neural_network')
            config: Configuración del proyecto
        """
        self.model_type = model_type
        self.model = None
        self.history = None
        self.config = config if config else load_config()
        
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrena un modelo SVM.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
        """
        print(f"\n🔧 Entrenando SVM...")
        
        # Obtener parámetros de configuración
        svm_config = self.config.get('model', {}).get('svm', {})
        
        self.model = SVC(
            kernel=svm_config.get('kernel', 'rbf'),
            C=svm_config.get('C', 1.0),
            gamma=svm_config.get('gamma', 'scale'),
            probability=svm_config.get('probability', True),
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("✅ SVM entrenado")
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Entrena un modelo Random Forest.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
        """
        print(f"\n🔧 Entrenando Random Forest...")
        
        # Obtener parámetros de configuración
        rf_config = self.config.get('model', {}).get('random_forest', {})
        
        self.model = RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 20),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=42,
            n_jobs=rf_config.get('n_jobs', -1)
        )
        self.model.fit(X_train, y_train)
        print("✅ Random Forest entrenado")
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray = None, y_val: np.ndarray = None,
                            epochs: int = None, batch_size: int = None) -> None:
        """
        Entrena una red neuronal.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            epochs: Número de épocas (usa config si no se especifica)
            batch_size: Tamaño del lote (usa config si no se especifica)
        """
        print(f"\n🔧 Entrenando Red Neuronal...")
        
        # Obtener parámetros de configuración
        nn_config = self.config.get('model', {}).get('neural_network', {})
        
        if epochs is None:
            epochs = nn_config.get('epochs', 50)
        if batch_size is None:
            batch_size = nn_config.get('batch_size', 32)
        
        early_stopping_patience = nn_config.get('early_stopping_patience', 10)
        reduce_lr_patience = nn_config.get('reduce_lr_patience', 5)
        
        self.model = create_neural_network(X_train.shape[1], self.config)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Red Neuronal entrenada")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> None:
        """
        Entrena el modelo según el tipo especificado.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Etiquetas de entrenamiento
            **kwargs: Argumentos adicionales para el entrenamiento
        """
        if self.model_type == 'svm':
            self.train_svm(X_train, y_train)
        elif self.model_type == 'random_forest':
            self.train_random_forest(X_train, y_train)
        elif self.model_type == 'neural_network':
            self.train_neural_network(X_train, y_train, **kwargs)
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones.
        
        Args:
            X: Características a predecir
            
        Returns:
            np.ndarray: Predicciones
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        if self.model_type == 'neural_network':
            predictions = self.model.predict(X, verbose=0)
            return (predictions > 0.5).astype(int).flatten()
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones probabilísticas.
        
        Args:
            X: Características a predecir
            
        Returns:
            np.ndarray: Probabilidades
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado")
        
        if self.model_type == 'neural_network':
            return self.model.predict(X, verbose=0)
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo en el conjunto de prueba.
        
        Args:
            X_test: Características de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Guarda el modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'neural_network':
            self.model.save(path)
        else:
            joblib.dump(self.model, path)
        
        print(f"✅ Modelo guardado en: {path}")
    
    def load(self, path: str) -> None:
        """
        Carga un modelo previamente guardado.
        
        Args:
            path: Ruta del modelo a cargar
        """
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(path)
        else:
            self.model = joblib.load(path)
        
        print(f"✅ Modelo cargado desde: {path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: str = None) -> None:
    """
    Genera y guarda la matriz de confusión.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        save_path: Ruta donde guardar la figura
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Placa', 'Placa'],
                yticklabels=['No Placa', 'Placa'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada en: {save_path}")
    
    plt.close()


def plot_training_history(history, save_path: str = None) -> None:
    """
    Grafica el historial de entrenamiento de la red neuronal.
    
    Args:
        history: Historial de entrenamiento de Keras
        save_path: Ruta donde guardar la figura
    """
    if history is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Pérdida durante el Entrenamiento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Precisión durante el Entrenamiento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Historial de entrenamiento guardado en: {save_path}")
    
    plt.close()


def train_and_evaluate_all_models(train_features: Dict, test_features: Dict,
                                   y_train: np.ndarray, y_test: np.ndarray,
                                   config: dict) -> Dict:
    """
    Entrena y evalúa todos los modelos con todas las combinaciones de características.
    
    Args:
        train_features: Diccionario con características de entrenamiento
        test_features: Diccionario con características de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        config: Configuración del proyecto
        
    Returns:
        dict: Diccionario con todos los modelos entrenados y sus métricas
    """
    print("\n" + "="*60)
    print("🚀 ENTRENANDO Y EVALUANDO MODELOS")
    print("="*60)
    
    results = {}
    models_dict = {}
    
    feature_types = ['hog', 'brisk']
    model_types = ['svm', 'random_forest', 'neural_network']
    
    for feature_type in feature_types:
        print(f"\n{'='*60}")
        print(f"📊 Usando características: {feature_type.upper()}")
        print(f"{'='*60}")
        
        X_train = train_features[feature_type]
        X_test = test_features[feature_type]
        
        for model_type in model_types:
            print(f"\n--- {model_type.upper().replace('_', ' ')} ---")
            
            # Crear y entrenar modelo con configuración
            classifier = PlateClassifier(model_type=model_type, config=config)
            
            if model_type == 'neural_network':
                # Usar configuración para validación
                nn_config = config.get('model', {}).get('neural_network', {})
                val_split = nn_config.get('validation_split', 0.2)
                
                val_split_idx = int((1 - val_split) * len(X_train))
                X_train_nn, X_val_nn = X_train[:val_split_idx], X_train[val_split_idx:]
                y_train_nn, y_val_nn = y_train[:val_split_idx], y_train[val_split_idx:]
                
                classifier.train(X_train_nn, y_train_nn, 
                               X_val=X_val_nn, y_val=y_val_nn)
            else:
                classifier.train(X_train, y_train)
            
            # Evaluar
            metrics = classifier.evaluate(X_test, y_test)
            
            # Imprimir métricas
            print(f"\n📊 Métricas de evaluación:")
            for metric_name, value in metrics.items():
                print(f"   • {metric_name.capitalize()}: {value:.4f}")
            
            # Guardar modelo
            model_name = f"{model_type}_{feature_type}"
            model_path = os.path.join(
                config['model']['save_path'], 
                f"{model_name}.{'h5' if model_type == 'neural_network' else 'pkl'}"
            )
            classifier.save(model_path)
            
            # Guardar matriz de confusión
            y_pred = classifier.predict(X_test)
            cm_path = os.path.join(
                config['evaluation']['results_path'],
                f"confusion_matrix_{model_name}.png"
            )
            plot_confusion_matrix(y_test, y_pred, cm_path)
            
            # Guardar historial de entrenamiento (solo NN)
            if model_type == 'neural_network' and classifier.history:
                history_path = os.path.join(
                    config['evaluation']['results_path'],
                    f"training_history_{model_name}.png"
                )
                plot_training_history(classifier.history, history_path)
            
            # Guardar resultados
            results[model_name] = {
                'model_type': model_type,
                'feature_type': feature_type,
                'metrics': metrics,
                'model_path': model_path
            }
            
            models_dict[model_name] = classifier
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE RESULTADOS")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        for metric, value in result['metrics'].items():
            print(f"   • {metric}: {value:.4f}")
    
    # Encontrar mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['metrics']['f1_score'])
    print(f"\n🏆 MEJOR MODELO: {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['metrics']['f1_score']:.4f}")
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    
    return results, models_dict


if __name__ == "__main__":
    from preprocessing import prepare_dataset
    from feature_extraction import extract_all_features
    
    # Cargar configuración
    config = load_config()
    
    # Preparar dataset
    X_train, X_test, y_train, y_test = prepare_dataset(config)
    
    # Extraer características
    train_features, test_features = extract_all_features(X_train, X_test, config)
    
    # Entrenar y evaluar modelos
    results, models = train_and_evaluate_all_models(
        train_features, test_features,
        y_train, y_test,
        config
    )
