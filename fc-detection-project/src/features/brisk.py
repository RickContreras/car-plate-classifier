"""
BRISK (Binary Robust Invariant Scalable Keypoints) feature extractor.
"""

import numpy as np
import cv2
from .base import FeatureExtractor


class BRISKFeatureExtractor(FeatureExtractor):
    """
    Extract BRISK features from images.
    
    BRISK is a feature detector and descriptor that is rotation and scale invariant,
    making it robust for various image conditions.
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
        Initialize BRISK feature extractor.
        
        Args:
            n_keypoints: Number of keypoints to extract (will pad or truncate to this)
            target_size: Target size (width, height) to resize images
            threshold: AGAST detection threshold
            octaves: Detection octaves
            pattern_scale: Scale of the pattern used for sampling
        """
        super().__init__(name="BRISK")
        
        self.n_keypoints = n_keypoints
        self.target_size = target_size
        self.threshold = threshold
        self.octaves = octaves
        self.pattern_scale = pattern_scale
        
        # Initialize BRISK detector
        self.brisk = cv2.BRISK_create(
            thresh=self.threshold,
            octaves=self.octaves,
            patternScale=self.pattern_scale
        )
        
        # BRISK descriptor is 64 bytes (512 bits)
        self.descriptor_size = 64
        self._feature_dim = self.n_keypoints * self.descriptor_size
        self._is_fitted = True
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract BRISK features from an image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            BRISK feature vector
        """
        # Preprocess image
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=True
        )
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.brisk.detectAndCompute(processed, None)
        
        # Handle case where no keypoints are detected
        if descriptors is None or len(descriptors) == 0:
            # Return zero vector
            return np.zeros(self._feature_dim, dtype=np.float32)
        
        # Normalize to n_keypoints
        if len(descriptors) < self.n_keypoints:
            # Pad with zeros
            padding = np.zeros(
                (self.n_keypoints - len(descriptors), self.descriptor_size),
                dtype=np.uint8
            )
            descriptors = np.vstack([descriptors, padding])
        elif len(descriptors) > self.n_keypoints:
            # Take top n_keypoints based on response strength
            responses = [kp.response for kp in keypoints]
            top_indices = np.argsort(responses)[-self.n_keypoints:]
            descriptors = descriptors[top_indices]
        
        # Flatten to 1D vector and normalize to [0, 1]
        features = descriptors.flatten().astype(np.float32) / 255.0
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of BRISK features."""
        return self._feature_dim
    
    def visualize_keypoints(self, image: np.ndarray) -> np.ndarray:
        """
        Visualize detected keypoints on an image.
        
        Args:
            image: Input image
            
        Returns:
            Image with keypoints drawn
        """
        # Preprocess image
        processed = self.preprocess_image(
            image,
            target_size=self.target_size,
            grayscale=False  # Keep color for visualization
        )
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if len(processed.shape) == 3 else processed
        
        # Detect keypoints
        keypoints = self.brisk.detect(gray, None)
        
        # Take top keypoints
        if len(keypoints) > self.n_keypoints:
            responses = [kp.response for kp in keypoints]
            top_indices = np.argsort(responses)[-self.n_keypoints:]
            keypoints = [keypoints[i] for i in top_indices]
        
        # Draw keypoints
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
        """Get configuration dictionary."""
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
