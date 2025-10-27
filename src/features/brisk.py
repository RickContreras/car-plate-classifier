"""
Extractor de características BRISK (Binary Robust Invariant Scalable Keypoints).
"""

import numpy as np
import cv2
from .base import FeatureExtractor


class BRISKFeatureExtractor(FeatureExtractor):
    """
    Extraer características BRISK de imágenes.
    
    BRISK es un detector y descriptor de características que es invariante a rotación y escala,
    haciéndolo robusto para varias condiciones de imagen.
    """
    
    def __init__(
        self,
        n_keypoints: int = 512,
        target_size: tuple = (200, 200),
        threshold: int = 30,
        octaves: int = 3,
        pattern_scale: float = 1.0
    ):
        """
        Inicializar extractor de características BRISK.
        
        Args:
            n_keypoints: Número de keypoints a extraer (se rellenará o truncará a esto)
            target_size: Tamaño objetivo (ancho, alto) para redimensionar imágenes
            threshold: Umbral de detección AGAST
            octaves: Octavas de detección
            pattern_scale: Escala del patrón usado para muestreo
        """
        super().__init__(name="BRISK")
        
        self.n_keypoints = n_keypoints
        self.target_size = target_size
        self.threshold = threshold
        self.octaves = octaves
        self.pattern_scale = pattern_scale
        
        # Inicializar detector BRISK
        self.brisk = cv2.BRISK_create(
            thresh=self.threshold,
            octaves=self.octaves,
            patternScale=self.pattern_scale
        )
        
        # El descriptor BRISK es de 64 bytes (512 bits)
        self.descriptor_size = 64
        self._feature_dim = self.n_keypoints * self.descriptor_size
        self._is_fitted = True
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extraer características BRISK de una imagen.
        
        Args:
            image: Imagen de entrada (BGR o escala de grises)
            
        Returns:
            Vector de características BRISK
        """
        # Preprocesar imagen
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=True
        )
        
        # Detectar keypoints y computar descriptores
        keypoints, descriptors = self.brisk.detectAndCompute(processed, None)
        
        # Manejar caso donde no se detectan keypoints
        if descriptors is None or len(descriptors) == 0:
            # Retornar vector de ceros
            return np.zeros(self._feature_dim, dtype=np.float32)
        
        # Normalizar a n_keypoints
        if len(descriptors) < self.n_keypoints:
            # Rellenar con ceros
            padding = np.zeros(
                (self.n_keypoints - len(descriptors), self.descriptor_size),
                dtype=np.uint8
            )
            descriptors = np.vstack([descriptors, padding])
        elif len(descriptors) > self.n_keypoints:
            # Tomar los n_keypoints superiores basados en la fuerza de respuesta
            responses = [kp.response for kp in keypoints]
            top_indices = np.argsort(responses)[-self.n_keypoints:]
            descriptors = descriptors[top_indices]
        
        # Aplanar a vector 1D y normalizar a [0, 1]
        features = descriptors.flatten().astype(np.float32) / 255.0
        
        return features
    
    def get_feature_dim(self) -> int:
        """Obtener la dimensionalidad de características BRISK."""
        return self._feature_dim
    
    def visualize_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Visualizar keypoints detectados en una imagen.
        
        Args:
            image: Imagen de entrada
            
        Returns:
            Imagen con keypoints dibujados
        """
        # Preprocesar imagen
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=False  # Mantener color para visualización
        )
        
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        
        # Detectar keypoints
        keypoints = self.brisk.detect(gray, None)
        
        # Tomar los keypoints superiores
        if len(keypoints) > self.n_keypoints:
            responses = [kp.response for kp in keypoints]
            top_indices = np.argsort(responses)[-self.n_keypoints:]
            keypoints = [keypoints[i] for i in top_indices]
        
        # Dibujar keypoints
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        img_keypoints = cv2.drawKeypoints(
            processed,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return img_keypoints
    
    def get_config(self) -> dict:
        """Obtener diccionario de configuración."""
        return {
            'name': self.name,
            'n_keypoints': self.n_keypoints,
            'target_size': self.target_size,
            'threshold': self.threshold,
            'octaves': self.octaves,
            'pattern_scale': self.pattern_scale,
            'descriptor_size': self.descriptor_size,
            'feature_dim': self._feature_dim
        }
