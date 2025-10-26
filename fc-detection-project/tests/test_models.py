"""
Tests for neural network models.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FCNetwork


class TestFCNetwork(unittest.TestCase):
    """Test Fully Connected Network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 8100
        self.model = FCNetwork(
            input_dim=self.input_dim,
            architecture=[512, 256, 128, 64, 4],
            use_batch_norm=True
        )
        self.model.compile(learning_rate=0.001)
    
    def test_model_creation(self):
        """Test model creation."""
        keras_model = self.model.get_model()
        
        self.assertIsNotNone(keras_model)
        self.assertEqual(keras_model.input_shape[1], self.input_dim)
        self.assertEqual(keras_model.output_shape[1], 4)
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Create dummy input
        X = np.random.randn(5, self.input_dim).astype(np.float32)
        
        # Predict
        predictions = self.model.get_model().predict(X, verbose=0)
        
        self.assertEqual(predictions.shape, (5, 4))
        # Output should be in [0, 1] due to sigmoid activation
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.model.get_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('input_dim', config)
        self.assertIn('architecture', config)
        self.assertIn('total_params', config)


if __name__ == '__main__':
    unittest.main()
