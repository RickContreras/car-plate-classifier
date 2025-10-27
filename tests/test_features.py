"""
Tests para extractores de características.
"""

import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import HOGFeatureExtractor, BRISKFeatureExtractor


class TestHOGFeatureExtractor(unittest.TestCase):
    """Test para extractor de características HOG."""
    
    def setUp(self):
        """Configurar fixtures de test."""
        self.extractor = HOGFeatureExtractor()
        self.test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    def test_extract_features(self):
        """Test para extracción de características."""
        features = self.extractor.extract(self.test_image)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features.shape), 1)  # Vector 1D
    
    def test_feature_dimension(self):
        """Test para consistencia de dimensión de características."""
        features1 = self.extractor.extract(self.test_image)
        features2 = self.extractor.extract(self.test_image)
        
        self.assertEqual(len(features1), len(features2))
        self.assertEqual(len(features1), self.extractor.get_feature_dim())
    
    def test_extract_batch(self):
        """Test para extracción de características en lote."""
        images = [self.test_image for _ in range(5)]
        features = self.extractor.extract_batch(images)
        
        self.assertEqual(features.shape[0], 5)
        self.assertEqual(features.shape[1], self.extractor.get_feature_dim())
    
    def test_get_config(self):
        """Test para obtención de configuración."""
        config = self.extractor.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('orientations', config)
        self.assertIn('feature_dim', config)


class TestBRISKFeatureExtractor(unittest.TestCase):
    """Test para extractor de características BRISK."""
    
    def setUp(self):
        """Configurar fixtures de test."""
        self.extractor = BRISKFeatureExtractor(n_keypoints=512)
        self.test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    def test_extract_features(self):
        """Test para extracción de características."""
        features = self.extractor.extract(self.test_image)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features.shape), 1)  # Vector 1D
    
    def test_feature_dimension(self):
        """Test para dimensión de características."""
        features = self.extractor.extract(self.test_image)
        expected_dim = 512 * 64  # n_keypoints * descriptor_size
        
        self.assertEqual(len(features), expected_dim)
        self.assertEqual(len(features), self.extractor.get_feature_dim())
    
    def test_extract_batch(self):
        """Test para extracción de características en lote."""
        images = [self.test_image for _ in range(3)]
        features = self.extractor.extract_batch(images)
        
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features.shape[1], self.extractor.get_feature_dim())
    
    def test_get_config(self):
        """Test para obtención de configuración."""
        config = self.extractor.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('n_keypoints', config)
        self.assertIn('feature_dim', config)


if __name__ == '__main__':
    unittest.main()
