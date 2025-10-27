"""
Tests para utilidades de datos y dataset.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DetectionDataset, normalize_bbox, denormalize_bbox


class TestBBoxUtils(unittest.TestCase):
    """Test para utilidades de bounding box."""
    
    def test_normalize_bbox(self):
        """Test para normalización de bbox."""
        bbox = (100, 150, 300, 250)  # xmin, ymin, xmax, ymax
        img_width, img_height = 640, 480
        
        norm_bbox = normalize_bbox(bbox, img_width, img_height)
        
        # Verificar que todos los valores estén en [0, 1]
        self.assertTrue(all(0 <= v <= 1 for v in norm_bbox))
        
        # Verificar que retorne 4 valores
        self.assertEqual(len(norm_bbox), 4)
    
    def test_denormalize_bbox(self):
        """Test para desnormalización de bbox."""
        norm_bbox = (0.5, 0.5, 0.3, 0.2)  # x_center, y_center, width, height
        img_width, img_height = 640, 480
        
        bbox = denormalize_bbox(norm_bbox, img_width, img_height)
        
        # Verificar que retorne 4 enteros
        self.assertEqual(len(bbox), 4)
        self.assertTrue(all(isinstance(v, int) for v in bbox))
        
        # Verificar que los valores estén dentro de los límites de la imagen
        xmin, ymin, xmax, ymax = bbox
        self.assertTrue(0 <= xmin < img_width)
        self.assertTrue(0 <= ymin < img_height)
        self.assertTrue(0 <= xmax <= img_width)
        self.assertTrue(0 <= ymax <= img_height)
    
    def test_normalize_denormalize_roundtrip(self):
        """Test para verificar que normalize -> denormalize preserva el bbox."""
        bbox = (100, 150, 300, 250)
        img_width, img_height = 640, 480
        
        norm_bbox = normalize_bbox(bbox, img_width, img_height)
        recovered_bbox = denormalize_bbox(norm_bbox, img_width, img_height)
        
        # Debería ser aproximadamente igual (con error de redondeo)
        for orig, recovered in zip(bbox, recovered_bbox):
            self.assertAlmostEqual(orig, recovered, delta=2)


class TestDetectionDataset(unittest.TestCase):
    """Test para clase DetectionDataset."""
    
    def setUp(self):
        """Configurar fixtures de test."""
        self.n_samples = 100
        self.feature_dim = 8100
        
        self.features = np.random.randn(self.n_samples, self.feature_dim).astype(np.float32)
        self.bboxes = np.random.rand(self.n_samples, 4).astype(np.float32)
        
        self.dataset = DetectionDataset(
            features=self.features,
            bboxes=self.bboxes
        )
    
    def test_dataset_creation(self):
        """Test para creación de dataset."""
        self.assertEqual(len(self.dataset), self.n_samples)
        self.assertEqual(self.dataset.features.shape, (self.n_samples, self.feature_dim))
        self.assertEqual(self.dataset.bboxes.shape, (self.n_samples, 4))
    
    def test_dataset_getitem(self):
        """Test para indexación de dataset."""
        features, bbox = self.dataset[0]
        
        self.assertEqual(features.shape, (self.feature_dim,))
        self.assertEqual(bbox.shape, (4,))
    
    def test_dataset_split(self):
        """Test para división de dataset."""
        train_dataset, val_dataset = self.dataset.split(train_ratio=0.8, shuffle=True, seed=42)
        
        # Verificar tamaños
        expected_train = int(0.8 * self.n_samples)
        expected_val = self.n_samples - expected_train
        
        self.assertEqual(len(train_dataset), expected_train)
        self.assertEqual(len(val_dataset), expected_val)
        
        # Verificar que las muestras totales coincidan
        self.assertEqual(len(train_dataset) + len(val_dataset), self.n_samples)


if __name__ == '__main__':
    unittest.main()
