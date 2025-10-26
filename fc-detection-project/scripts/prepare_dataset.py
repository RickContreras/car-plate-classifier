#!/usr/bin/env python3
"""
Script to prepare dataset for detection training.

Extracts features (HOG or BRISK) from images and saves them with bounding box labels.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.data.utils import parse_pascal_voc, normalize_bbox, DetectionDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_dataset(
    images_dir: str,
    annotations_dir: str,
    feature_type: str,
    config: dict,
    output_path: str
):
    """
    Prepare dataset with features and labels.
    
    Args:
        images_dir: Directory with images
        annotations_dir: Directory with XML annotations
        feature_type: 'hog' or 'brisk'
        config: Configuration dictionary
        output_path: Path to save processed dataset
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    
    # Initialize feature extractor
    feature_params = config['feature_extractor']['params']
    
    if feature_type == 'hog':
        extractor = HOGFeatureExtractor(**feature_params)
    elif feature_type == 'brisk':
        extractor = BRISKFeatureExtractor(**feature_params)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    print(f"\n{'='*60}")
    print(f"Preparing {feature_type.upper()} dataset")
    print(f"{'='*60}")
    print(f"Feature extractor: {extractor}")
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"Images directory: {images_dir}")
    print(f"Annotations directory: {annotations_dir}")
    print(f"{'='*60}\n")
    
    # Get all XML files
    xml_files = list(annotations_path.glob('*.xml'))
    
    if len(xml_files) == 0:
        print(f"ERROR: No XML files found in {annotations_dir}")
        return
    
    print(f"Found {len(xml_files)} annotation files")
    
    features_list = []
    bboxes_list = []
    image_paths_list = []
    skipped = 0
    
    # Process each annotation
    for xml_file in tqdm(xml_files, desc="Processing images"):
        try:
            # Parse XML
            filename, bboxes = parse_pascal_voc(str(xml_file))
            
            # Skip if no bboxes
            if len(bboxes) == 0:
                skipped += 1
                continue
            
            # Load image
            image_path = images_path / filename
            if not image_path.exists():
                # Try with different extension
                image_path = images_path / (Path(filename).stem + '.jpg')
                if not image_path.exists():
                    image_path = images_path / (Path(filename).stem + '.png')
            
            if not image_path.exists():
                print(f"Warning: Image not found: {filename}")
                skipped += 1
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Failed to load image: {filename}")
                skipped += 1
                continue
            
            height, width = image.shape[:2]
            
            # Extract features
            features = extractor.extract(image)
            
            # Process each bbox (use only the first one for single object detection)
            bbox = bboxes[0]
            normalized_bbox = normalize_bbox(bbox, width, height)
            
            # Add to lists
            features_list.append(features)
            bboxes_list.append(normalized_bbox)
            image_paths_list.append(str(image_path))
            
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessed {len(features_list)} samples")
    print(f"Skipped {skipped} files")
    
    if len(features_list) == 0:
        print("ERROR: No samples processed")
        return
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    bboxes_array = np.array(bboxes_list)
    
    print(f"\nFeatures shape: {features_array.shape}")
    print(f"Bboxes shape: {bboxes_array.shape}")
    
    # Create dataset
    dataset = DetectionDataset(
        features=features_array,
        bboxes=bboxes_array,
        image_paths=image_paths_list
    )
    
    # Save dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(output_file))
    
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Prepare detection dataset')
    parser.add_argument('--images', type=str, required=True, help='Images directory')
    parser.add_argument('--annotations', type=str, required=True, help='Annotations directory')
    parser.add_argument('--feature-type', type=str, required=True, choices=['hog', 'brisk'], help='Feature type')
    parser.add_argument('--config', type=str, help='Config file path')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Use default config
        config_file = f"configs/{args.feature_type}_config.yaml"
        if Path(config_file).exists():
            config = load_config(config_file)
        else:
            print(f"ERROR: Config file not found: {config_file}")
            print("Please provide --config or create default config file")
            return
    
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"data/processed/detection_{args.feature_type}.pkl"
    
    # Prepare dataset
    prepare_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        feature_type=args.feature_type,
        config=config,
        output_path=output_path
    )


if __name__ == '__main__':
    main()
