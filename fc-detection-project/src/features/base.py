"""
Base class for feature extractors.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import cv2


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    def __init__(self, name: str = "base"):
        """
        Initialize feature extractor.
        
        Args:
            name: Name of the feature extractor
        """
        self.name = name
        self._is_fitted = False
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of the feature vector.
        
        Returns:
            Integer representing feature dimension
        """
        pass
    
    def preprocess_image(
        self, 
        image: np.ndarray, 
        target_size: tuple = None,
        grayscale: bool = True
    ) -> np.ndarray:
        """
        Preprocess image before feature extraction.
        
        Args:
            image: Input image
            target_size: Tuple (width, height) to resize image
            grayscale: Whether to convert to grayscale
            
        Returns:
            Preprocessed image
        """
        # Resize if target size is specified
        if target_size is not None:
            image = cv2.resize(image, target_size)
        
        # Convert to grayscale if needed
        if grayscale and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Array of shape (n_images, feature_dim)
        """
        features = []
        for image in images:
            feature = self.extract(image)
            features.append(feature)
        
        return np.array(features)
    
    def __repr__(self) -> str:
        """String representation of the feature extractor."""
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.get_feature_dim()})"
