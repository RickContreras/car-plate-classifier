"""
Extractor de características HOG (Histogram of Oriented Gradients).
"""

import numpy as np
import cv2
from skimage.feature import hog
from .base import FeatureExtractor


class HOGFeatureExtractor(FeatureExtractor):
    """
    Extraer características HOG de imágenes.
    
    HOG (Histogram of Oriented Gradients) captura la estructura de bordes y gradientes,
    haciéndolo efectivo para tareas de detección y localización de objetos.
    """
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (3, 3),
        block_norm: str = 'L2-Hys',
        target_size: tuple = (200, 200),
        transform_sqrt: bool = True,
        feature_vector: bool = True
    ):
        """
        Inicializar extractor de características HOG.
        
        Args:
            orientations: Número de bins de orientación
            pixels_per_cell: Tamaño (en píxeles) de una celda
            cells_per_block: Número de celdas en cada bloque
            block_norm: Método de normalización de bloque ('L1', 'L1-sqrt', 'L2', 'L2-Hys')
            target_size: Tamaño objetivo (ancho, alto) para redimensionar imágenes
            transform_sqrt: Aplicar compresión de ley de potencia para normalizar la imagen
            feature_vector: Retornar características como un solo vector
        """
        super().__init__(name="HOG")
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.target_size = target_size
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        
        # Calcular dimensión de características
        self._calculate_feature_dim()
        self._is_fitted = True
    
    def _calculate_feature_dim(self):
        """Calcular la dimensionalidad de las características HOG."""
        # Crear una imagen ficticia para obtener la dimensión de características
        dummy_image = np.zeros((self.target_size[1], self.target_size[0]))
        dummy_features = hog(
            dummy_image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            feature_vector=self.feature_vector
        )
        self._feature_dim = len(dummy_features)
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extraer características HOG de una imagen.
        
        Args:
            image: Imagen de entrada (BGR o escala de grises)
            
        Returns:
            Vector de características HOG
        """
        # Preprocesar imagen
        processed = self.preprocess_image(
            image, 
            target_size=self.target_size,
            grayscale=True
        )
        
        # Extraer características HOG
        features = hog(
            processed,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            feature_vector=self.feature_vector
        )
        
        return features.astype(np.float32)
    
    def get_feature_dim(self) -> int:
        """Obtener la dimensionalidad de características HOG."""
        return self._feature_dim
    
    def visualize_hog(self, image: np.ndarray) -> np.ndarray:
        """
        Visualizar características HOG para una imagen.
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Imagen de visualización HOG
        """
        # Preprocesar imagen
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=True
        )
        
        # Extraer HOG con visualización
        _, hog_image = hog(
            processed,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=True,
            feature_vector=self.feature_vector
        )
        
        return hog_image
    
    def get_config(self) -> dict:
        """Obtener diccionario de configuración."""
        return {
            'name': self.name,
            'orientations': self.orientations,
            'pixels_per_cell': self.pixels_per_cell,
            'cells_per_block': self.cells_per_block,
            'block_norm': self.block_norm,
            'target_size': self.target_size,
            'transform_sqrt': self.transform_sqrt,
            'feature_dim': self._feature_dim
        }
