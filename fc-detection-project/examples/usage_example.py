"""
Example usage of FC Detection Project

This script demonstrates how to use all components of the project.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.models import FCNetwork
from src.data import DetectionDataset, normalize_bbox, denormalize_bbox
from src.training import Trainer, get_callbacks
from src.evaluation import MetricsCalculator


def example_feature_extraction():
    """Example: Extract features from an image."""
    print("\n" + "="*60)
    print("Example 1: Feature Extraction")
    print("="*60)
    
    # Create a dummy image
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    
    # HOG features
    hog_extractor = HOGFeatureExtractor()
    hog_features = hog_extractor.extract(image)
    print(f"HOG features shape: {hog_features.shape}")
    print(f"HOG feature dim: {hog_extractor.get_feature_dim()}")
    
    # BRISK features
    brisk_extractor = BRISKFeatureExtractor(n_keypoints=512)
    brisk_features = brisk_extractor.extract(image)
    print(f"BRISK features shape: {brisk_features.shape}")
    print(f"BRISK feature dim: {brisk_extractor.get_feature_dim()}")


def example_model_creation():
    """Example: Create and compile a model."""
    print("\n" + "="*60)
    print("Example 2: Model Creation")
    print("="*60)
    
    # Create model
    model = FCNetwork(
        input_dim=8100,
        architecture=[512, 256, 128, 64, 4],
        use_batch_norm=True
    )
    
    # Compile
    model.compile(learning_rate=0.001)
    
    # Show summary
    model.summary()
    
    # Test prediction
    X = np.random.randn(5, 8100).astype(np.float32)
    predictions = model.get_model().predict(X, verbose=0)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample prediction: {predictions[0]}")


def example_dataset_operations():
    """Example: Dataset operations."""
    print("\n" + "="*60)
    print("Example 3: Dataset Operations")
    print("="*60)
    
    # Create dummy data
    n_samples = 100
    features = np.random.randn(n_samples, 8100).astype(np.float32)
    bboxes = np.random.rand(n_samples, 4).astype(np.float32)
    
    # Create dataset
    dataset = DetectionDataset(features=features, bboxes=bboxes)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    train_ds, val_ds = dataset.split(train_ratio=0.8, seed=42)
    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    
    # Get a sample
    sample_features, sample_bbox = dataset[0]
    print(f"\nSample features shape: {sample_features.shape}")
    print(f"Sample bbox: {sample_bbox}")


def example_bbox_utils():
    """Example: Bounding box utilities."""
    print("\n" + "="*60)
    print("Example 4: Bounding Box Utilities")
    print("="*60)
    
    # Original bbox in pixels
    bbox_pixels = (100, 150, 300, 250)  # xmin, ymin, xmax, ymax
    img_width, img_height = 640, 480
    
    print(f"Original bbox (pixels): {bbox_pixels}")
    print(f"Image size: {img_width}x{img_height}")
    
    # Normalize
    bbox_norm = normalize_bbox(bbox_pixels, img_width, img_height)
    print(f"Normalized bbox: {bbox_norm}")
    
    # Denormalize back
    bbox_recovered = denormalize_bbox(bbox_norm, img_width, img_height)
    print(f"Recovered bbox (pixels): {bbox_recovered}")


def example_metrics():
    """Example: Calculate metrics."""
    print("\n" + "="*60)
    print("Example 5: Metrics Calculation")
    print("="*60)
    
    # Create dummy predictions and ground truth
    n_samples = 50
    y_true = np.random.rand(n_samples, 4).astype(np.float32)
    y_pred = y_true + np.random.randn(n_samples, 4).astype(np.float32) * 0.1
    y_pred = np.clip(y_pred, 0, 1)
    
    # Calculate metrics
    metrics_calc = MetricsCalculator(iou_thresholds=[0.5, 0.75, 0.9])
    metrics_calc.update(y_true, y_pred)
    metrics = metrics_calc.compute()
    
    # Print metrics
    metrics_calc.print_metrics(metrics)


def example_training_setup():
    """Example: Set up training."""
    print("\n" + "="*60)
    print("Example 6: Training Setup")
    print("="*60)
    
    # Create dummy data
    X_train = np.random.randn(80, 8100).astype(np.float32)
    y_train = np.random.rand(80, 4).astype(np.float32)
    X_val = np.random.randn(20, 8100).astype(np.float32)
    y_val = np.random.rand(20, 4).astype(np.float32)
    
    # Create model
    model = FCNetwork(input_dim=8100)
    model.compile(learning_rate=0.001)
    
    # Get callbacks
    callbacks = get_callbacks(
        model_name='example_model',
        patience=5,
        reduce_lr_patience=3
    )
    
    print(f"Callbacks: {[type(cb).__name__ for cb in callbacks]}")
    
    # Create trainer
    trainer = Trainer(
        model=model.get_model(),
        save_dir='models',
        name='example_model'
    )
    
    print("\nTrainer created successfully")
    print("Ready to train with: trainer.train(X_train, y_train, X_val, y_val, ...)")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print(" FC Detection Project - Usage Examples")
    print("="*70)
    
    # Run examples
    example_feature_extraction()
    example_model_creation()
    example_dataset_operations()
    example_bbox_utils()
    example_metrics()
    example_training_setup()
    
    print("\n" + "="*70)
    print(" All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
