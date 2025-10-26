"""
Módulo de extracción de características HOG y BRISK para clasificación de placas.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from typing import Tuple, List
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carga la configuración desde el archivo YAML."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


class HOGFeatureExtractor:
    """Extractor de características HOG (Histogram of Oriented Gradients)."""
    
    def __init__(self, config: dict):
        """
        Inicializa el extractor HOG.
        
        Args:
            config: Configuración del proyecto
        """
        self.orientations = config['features']['hog']['orientations']
        self.pixels_per_cell = tuple(config['features']['hog']['pixels_per_cell'])
        self.cells_per_block = tuple(config['features']['hog']['cells_per_block'])
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características HOG de una imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            np.ndarray: Vector de características HOG
        """
        # Asegurarse de que la imagen esté en escala de grises
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Desnormalizar si está normalizada
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        
        return features
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extrae características HOG de un lote de imágenes.
        
        Args:
            images: Array de imágenes
            
        Returns:
            np.ndarray: Matriz de características (n_samples, n_features)
        """
        features_list = []
        
        for i, image in enumerate(images):
            if (i + 1) % 50 == 0:
                print(f"   Procesando imagen {i+1}/{len(images)}...")
            features = self.extract(image)
            features_list.append(features)
        
        return np.array(features_list)


class BRISKFeatureExtractor:
    """Extractor de características BRISK (Binary Robust Invariant Scalable Keypoints)."""
    
    def __init__(self, config: dict):
        """
        Inicializa el extractor BRISK.
        
        Args:
            config: Configuración del proyecto
        """
        self.threshold = config['features']['brisk']['threshold']
        self.octaves = config['features']['brisk']['octaves']
        self.pattern_scale = config['features']['brisk']['pattern_scale']
        self.brisk = cv2.BRISK_create(
            thresh=self.threshold,
            octaves=self.octaves,
            patternScale=self.pattern_scale
        )
        self.feature_size = 512  # Tamaño fijo para el vector de características
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae características BRISK de una imagen.
        
        Args:
            image: Imagen en escala de grises
            
        Returns:
            np.ndarray: Vector de características BRISK
        """
        # Asegurarse de que la imagen esté en escala de grises
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Desnormalizar si está normalizada
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Detectar keypoints y calcular descriptores
        keypoints, descriptors = self.brisk.detectAndCompute(image, None)
        
        # Si no se encuentran keypoints, devolver vector de ceros
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.feature_size)
        
        # Crear un vector de características de tamaño fijo
        # Estrategia: usar estadísticas de los descriptores
        features = self._create_fixed_size_descriptor(descriptors)
        
        return features
    
    def _create_fixed_size_descriptor(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Crea un vector de características de tamaño fijo a partir de descriptores BRISK.
        
        Args:
            descriptors: Descriptores BRISK variables
            
        Returns:
            np.ndarray: Vector de características de tamaño fijo
        """
        # Calcular estadísticas por columna
        mean = np.mean(descriptors, axis=0)  # 64 valores
        std = np.std(descriptors, axis=0)    # 64 valores
        max_vals = np.max(descriptors, axis=0)  # 64 valores
        min_vals = np.min(descriptors, axis=0)  # 64 valores
        
        # Concatenar estadísticas
        features = np.concatenate([mean, std, max_vals, min_vals])
        
        # Agregar información adicional
        num_keypoints = len(descriptors)
        features = np.append(features, num_keypoints)
        
        # Padding para llegar a feature_size
        if len(features) < self.feature_size:
            features = np.pad(features, (0, self.feature_size - len(features)))
        else:
            features = features[:self.feature_size]
        
        return features
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Extrae características BRISK de un lote de imágenes.
        
        Args:
            images: Array de imágenes
            
        Returns:
            np.ndarray: Matriz de características (n_samples, n_features)
        """
        features_list = []
        
        for i, image in enumerate(images):
            if (i + 1) % 50 == 0:
                print(f"   Procesando imagen {i+1}/{len(images)}...")
            features = self.extract(image)
            features_list.append(features)
        
        return np.array(features_list)


def extract_all_features(X_train: np.ndarray, X_test: np.ndarray, 
                         config: dict) -> Tuple[dict, dict]:
    """
    Extrae todas las características (HOG y BRISK) de los datasets.
    
    Args:
        X_train: Imágenes de entrenamiento
        X_test: Imágenes de prueba
        config: Configuración del proyecto
        
    Returns:
        Tuple: (train_features, test_features) donde cada uno es un dict con 'hog' y 'brisk'
    """
    print("\n" + "="*60)
    print("🔍 EXTRAYENDO CARACTERÍSTICAS")
    print("="*60)
    
    # Inicializar extractores
    hog_extractor = HOGFeatureExtractor(config)
    brisk_extractor = BRISKFeatureExtractor(config)
    
    # Extraer HOG
    print("\n📊 Extrayendo características HOG...")
    print(f"   Configuración: {hog_extractor.orientations} orientaciones, "
          f"{hog_extractor.pixels_per_cell} pixels/cell, "
          f"{hog_extractor.cells_per_block} cells/block")
    
    X_train_hog = hog_extractor.extract_batch(X_train)
    X_test_hog = hog_extractor.extract_batch(X_test)
    
    print(f"✅ HOG completado - Shape: {X_train_hog.shape}")
    
    # Extraer BRISK
    print("\n📊 Extrayendo características BRISK...")
    print(f"   Configuración: threshold={brisk_extractor.threshold}, "
          f"octaves={brisk_extractor.octaves}")
    
    X_train_brisk = brisk_extractor.extract_batch(X_train)
    X_test_brisk = brisk_extractor.extract_batch(X_test)
    
    print(f"✅ BRISK completado - Shape: {X_train_brisk.shape}")
    
    train_features = {
        'hog': X_train_hog,
        'brisk': X_train_brisk
    }
    
    test_features = {
        'hog': X_test_hog,
        'brisk': X_test_brisk
    }
    
    print("\n" + "="*60)
    print("✅ EXTRACCIÓN DE CARACTERÍSTICAS COMPLETADA")
    print("="*60)
    
    return train_features, test_features


if __name__ == "__main__":
    # Prueba del módulo
    from preprocessing import prepare_dataset
    
    config = load_config()
    X_train, X_test, y_train, y_test = prepare_dataset(config)
    train_features, test_features = extract_all_features(X_train, X_test, config)
    
    print(f"\n📊 Resumen de características:")
    print(f"   HOG - Train: {train_features['hog'].shape}, Test: {test_features['hog'].shape}")
    print(f"   BRISK - Train: {train_features['brisk'].shape}, Test: {test_features['brisk'].shape}")
