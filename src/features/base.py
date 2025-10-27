"""
Clase base para extractores de características.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import cv2


class FeatureExtractor(ABC):
    """Clase base abstracta para extractores de características."""
    
    def __init__(self, name: str = "base"):
        """
        Inicializar extractor de características.
        
        Args:
            name: Nombre del extractor de características
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extraer características de una imagen.
        
        Args:
            image: Imagen de entrada (BGR o escala de grises)
            
        Returns:
            Vector de características como array numpy
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Obtener la dimensionalidad del vector de características.
        
        Returns:
            Entero representando la dimensión de características
        """
        pass
    
    def preprocess_image(
        self, 
        image: np.ndarray, 
        target_size: tuple = None,
        grayscale: bool = True
    ) -> np.ndarray:
        """
        Preprocesar imagen antes de extracción de características.
        
        Args:
            image: Imagen de entrada
            target_size: Tupla (ancho, alto) para redimensionar imagen
            grayscale: Si se debe convertir a escala de grises
            
        Returns:
            Imagen preprocesada
        """
        # Redimensionar si se especifica un tamaño objetivo
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        # Convertir a escala de grises si es necesario
        if grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extraer características de un lote de imágenes.
        
        Args:
            images: Lista de imágenes
            
        Returns:
            Array de forma (n_images, feature_dim)
        """
        features = []
        for image in images:
            feature = self.extract(image)
            features.append(feature)
        
        return np.array(features)
    
    def __repr__(self) -> str:
        """Representación en cadena del extractor de características."""
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.get_feature_dim()})"
