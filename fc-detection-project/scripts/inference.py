#!/usr/bin/env python3
"""
Script to perform inference with trained detection models.
"""

import argparse
import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import os

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras
from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.data.utils import denormalize_bbox


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def perform_inference(
    model_path: str,
    image_path: str,
    feature_type: str,
    config: dict,
    output_path: str = None,
    show: bool = False
):
    """
    Perform inference on a single image.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        feature_type: 'hog' or 'brisk'
        config: Configuration dictionary
        output_path: Path to save result image
        show: Whether to display result
    """
    print(f"\n{'='*60}")
    print(f"Inference with {feature_type.upper()} Model")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✓ Model loaded")
    
    # Initialize feature extractor
    feature_params = config['feature_extractor']['params']
    
    if feature_type == 'hog':
        extractor = HOGFeatureExtractor(**feature_params)
    elif feature_type == 'brisk':
        extractor = BRISKFeatureExtractor(**feature_params)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    print(f"✓ Feature extractor initialized: {extractor}")
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"✓ Image loaded: {width}x{height}")
    
    # Extract features
    print("\nExtracting features...")
    features = extractor.extract(image)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    print(f"✓ Features extracted: {features.shape}")
    
    # Predict bounding box
    print("\nPredicting bounding box...")
    pred_bbox_norm = model.predict(features, verbose=0)[0]
    print(f"✓ Normalized bbox: [{pred_bbox_norm[0]:.3f}, {pred_bbox_norm[1]:.3f}, {pred_bbox_norm[2]:.3f}, {pred_bbox_norm[3]:.3f}]")
    
    # Denormalize bbox
    pred_bbox = denormalize_bbox(pred_bbox_norm, width, height)
    xmin, ymin, xmax, ymax = pred_bbox
    print(f"✓ Pixel bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
    
    # Draw bounding box
    result_image = image.copy()
    cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Add label
    label = f"{feature_type.upper()}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        result_image,
        (xmin, ymin - label_size[1] - 10),
        (xmin + label_size[0], ymin),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        result_image,
        label,
        (xmin, ymin - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )
    
    # Save result
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\n✓ Result saved to: {output_path}")
    
    # Show result
    if show:
        cv2.imshow('Detection Result', result_image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"{'='*60}\n")
    
    return result_image, pred_bbox


def main():
    parser = argparse.ArgumentParser(description='Perform inference with detection model')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--feature-type', type=str, required=True, choices=['hog', 'brisk'], help='Feature type')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--show', action='store_true', help='Display result')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config_file = f"configs/{args.feature_type}_config.yaml"
        if Path(config_file).exists():
            config = load_config(config_file)
        else:
            print(f"ERROR: Config file not found: {config_file}")
            return
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.image)
        output_path = f"results/{input_path.stem}_{args.feature_type}_detection{input_path.suffix}"
    
    # Check model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        return
    
    # Check image exists
    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return
    
    # Perform inference
    perform_inference(
        model_path=args.model,
        image_path=args.image,
        feature_type=args.feature_type,
        config=config,
        output_path=output_path,
        show=args.show
    )


if __name__ == '__main__':
    main()
