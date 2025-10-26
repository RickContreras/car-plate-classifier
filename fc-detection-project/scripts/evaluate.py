#!/usr/bin/env python3
"""
Script to evaluate trained detection models.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import json
import os

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras
from src.data import load_dataset
from src.evaluation import MetricsCalculator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(model_path: str, dataset_path: str, config: dict, feature_type: str):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset
        config: Configuration dictionary
        feature_type: 'hog' or 'brisk'
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {feature_type.upper()} Detection Model")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("✓ Model loaded")
    
    # Load dataset
    print(f"\nLoading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Split dataset to get test set
    data_config = config['data']
    train_dataset, test_dataset = dataset.split(
        train_ratio=data_config['train_ratio'],
        shuffle=data_config['shuffle'],
        seed=data_config.get('seed', 42)
    )
    
    X_test = test_dataset.features
    y_test = test_dataset.bboxes
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test, verbose=0)
    print("✓ Predictions complete")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics_calc = MetricsCalculator(
        iou_thresholds=config['evaluation']['iou_thresholds']
    )
    metrics_calc.update(y_test, y_pred)
    metrics = metrics_calc.compute()
    
    # Print metrics
    metrics_calc.print_metrics(metrics)
    
    # Save metrics to file
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = Path(model_path).stem
    metrics_path = results_dir / f"{model_name}_eval_metrics.json"
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    print("-" * 60)
    n_samples = min(5, len(y_test))
    for i in range(n_samples):
        true = y_test[i]
        pred = y_pred[i]
        iou = metrics_calc._calculate_iou(pred, true)
        
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: [{true[0]:.3f}, {true[1]:.3f}, {true[2]:.3f}, {true[3]:.3f}]")
        print(f"  Prediction:   [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}, {pred[3]:.3f}]")
        print(f"  IoU: {iou:.3f}")
    print("-" * 60)
    
    print(f"\n{'='*60}")
    print("Evaluation completed!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate detection model')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--dataset', type=str, help='Dataset file path')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--feature-type', type=str, choices=['hog', 'brisk'], help='Feature type')
    
    args = parser.parse_args()
    
    # Determine feature type
    if args.feature_type:
        feature_type = args.feature_type
    else:
        # Extract from model name
        model_name = Path(args.model).stem
        if 'hog' in model_name:
            feature_type = 'hog'
        elif 'brisk' in model_name:
            feature_type = 'brisk'
        else:
            print("ERROR: Cannot determine feature type. Use --feature-type")
            return
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config_file = f"configs/{feature_type}_config.yaml"
        if Path(config_file).exists():
            config = load_config(config_file)
        else:
            print(f"ERROR: Config file not found: {config_file}")
            return
    
    # Set dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = f"data/processed/detection_{feature_type}.pkl"
    
    # Check files exist
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return
    
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return
    
    # Evaluate model
    evaluate_model(args.model, dataset_path, config, feature_type)


if __name__ == '__main__':
    main()
