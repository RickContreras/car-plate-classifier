"""
HOG (Histogram of Oriented Gradients) feature extractor.
"""

import numpy as np
import cv2
from skimage.feature import hog
from .base import FeatureExtractor


class HOGFeatureExtractor(FeatureExtractor):
    """
    Extract HOG features from images.
    
    HOG (Histogram of Oriented Gradients) captures edge and gradient structure,
    making it effective for object detection and localization tasks.
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
        Initialize HOG feature extractor.
        
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size (in pixels) of a cell
            cells_per_block: Number of cells in each block
            block_norm: Block normalization method ('L1', 'L1-sqrt', 'L2', 'L2-Hys')
            target_size: Target size (width, height) to resize images
            transform_sqrt: Apply power law compression to normalize the image
            feature_vector: Return features as a single vector
        """
        super().__init__(name="HOG")
        
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.target_size = target_size
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector
        
        # Calculate feature dimension
        self._calculate_feature_dim()
        self._is_fitted = True
    
    def _calculate_feature_dim(self):
        """Calculate the dimensionality of HOG features."""
        # Create a dummy image to get feature dimension
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
        Extract HOG features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            HOG feature vector
        """
        # Preprocess image
        processed = self.preprocess_image(
            image, 
            target_size=self.target_size,
            grayscale=True
        )
        
        # Extract HOG features
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
        """Get the dimensionality of HOG features."""
        return self._feature_dim
    
    def visualize_hog(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize HOG features for an image.
        
        Args:
            image: Input image
            
        Returns:
            HOG visualization image
        """
        # Preprocess image
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=True
        )
        
        # Extract HOG with visualization
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
        """Get configuration dictionary."""
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
