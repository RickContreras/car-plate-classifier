"""
Script para preparar dataset para detección con Red Neuronal Completamente Conectada.

Extrae las coordenadas de bounding boxes de los XMLs y características HOG/BRISK.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import pickle
import yaml
from sklearn.model_selection import train_test_split

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor
from src.preprocessing import preprocess_image


def parse_xml_annotation(xml_path):
    """Parse XML annotation file and extract bounding box."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Get bounding box
    obj = root.find('object')
    if obj is None:
        return None
        
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    # Calculate center, width, height (normalized to 0-1)
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    
    return {
        'img_width': img_width,
        'img_height': img_height,
        'bbox_normalized': [x_center, y_center, width, height],  # normalized 0-1
        'bbox_absolute': [xmin, ymin, xmax, ymax]  # absolute pixels
    }


def extract_features(img_path, config, hog_extractor, brisk_extractor):
    """Extract HOG and BRISK features from image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    
    # Preprocess image
    processed = preprocess_image(img, config)
    
    # Extract HOG features
    hog_features = hog_extractor.extract(processed)
    
    # Extract BRISK features
    brisk_features = brisk_extractor.extract(processed)
    
    return hog_features, brisk_features


def prepare_detection_dataset(
    annotations_dir='data/raw/annotations',
    images_dir='data/raw/images',
    output_dir='data/detection_dataset',
    config_path='config/config.yaml',
    test_size=0.2,
    random_state=42
):
    """
    Prepare dataset for bounding box detection with fully connected neural network.
    Uses HOG and BRISK features instead of raw pixels.
    
    Args:
        annotations_dir: Directory containing XML files
        images_dir: Directory containing images
        output_dir: Output directory for processed data
        config_path: Path to configuration file
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
    """
    print("=" * 70)
    print("🔧 PREPARANDO DATASET PARA DETECCIÓN CON RED NEURONAL")
    print("=" * 70)
    print()
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize feature extractors
    print("⚙️  Inicializando extractores de características...")
    hog_extractor = HOGFeatureExtractor(config)
    brisk_extractor = BRISKFeatureExtractor(config)
    print("   ✅ Extractores inicializados")
    print()
    
    annotations_path = Path(annotations_dir)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    xml_files = list(annotations_path.glob('*.xml'))
    print(f"📁 Encontrados {len(xml_files)} archivos XML")
    print()
    
    # Prepare data
    X_hog_list = []     # HOG features
    X_brisk_list = []   # BRISK features
    y_boxes = []        # Bounding box coordinates (normalized)
    metadata = []       # Metadata for reference
    
    print("🔨 Procesando imágenes y extrayendo características...")
    processed = 0
    skipped = 0
    
    for xml_file in xml_files:
        # Parse XML
        annotation = parse_xml_annotation(xml_file)
        if annotation is None:
            skipped += 1
            continue
        
        # Find corresponding image
        img_name = xml_file.stem
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            potential_img = images_path / f"{img_name}{ext}"
            if potential_img.exists():
                img_file = potential_img
                break
        
        if img_file is None:
            skipped += 1
            continue
        
        # Extract features
        hog_features, brisk_features = extract_features(
            img_file, config, hog_extractor, brisk_extractor
        )
        
        if hog_features is None or brisk_features is None:
            skipped += 1
            continue
        
        # Store data
        X_hog_list.append(hog_features)
        X_brisk_list.append(brisk_features)
        y_boxes.append(annotation['bbox_normalized'])
        metadata.append({
            'filename': img_file.name,
            'img_width': annotation['img_width'],
            'img_height': annotation['img_height'],
            'bbox_absolute': annotation['bbox_absolute']
        })
        
        processed += 1
        
        if processed % 50 == 0:
            print(f"   Procesadas: {processed} imágenes...")
    
    print(f"\n✅ Procesamiento completado:")
    print(f"   • Imágenes procesadas: {processed}")
    print(f"   • Imágenes omitidas: {skipped}")
    print()
    
    # Convert to numpy arrays
    X_hog = np.array(X_hog_list, dtype=np.float32)
    X_brisk = np.array(X_brisk_list, dtype=np.float32)
    y = np.array(y_boxes, dtype=np.float32)
    
    print("📊 Dimensiones del dataset:")
    print(f"   • X_hog (características HOG): {X_hog.shape} ({X_hog.shape[1]} características)")
    print(f"   • X_brisk (características BRISK): {X_brisk.shape} ({X_brisk.shape[1]} características)")
    print(f"   • y (bounding boxes): {y.shape}")
    print()
    
    # Split into train/test for HOG
    X_hog_train, X_hog_test, y_hog_train, y_hog_test, meta_hog_train, meta_hog_test = train_test_split(
        X_hog, y, metadata,
        test_size=test_size,
        random_state=random_state
    )
    
    # Split into train/test for BRISK (using same split)
    X_brisk_train, X_brisk_test, y_brisk_train, y_brisk_test, meta_brisk_train, meta_brisk_test = train_test_split(
        X_brisk, y, metadata,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"📂 División train/test ({int((1-test_size)*100)}/{int(test_size*100)}):")
    print(f"   • Train: {len(X_hog_train)} imágenes")
    print(f"   • Test: {len(X_hog_test)} imágenes")
    print()
    
    # Save processed data for HOG
    print("💾 Guardando datos procesados...")
    
    data_hog = {
        'X_train': X_hog_train,
        'X_test': X_hog_test,
        'y_train': y_hog_train,
        'y_test': y_hog_test,
        'metadata_train': meta_hog_train,
        'metadata_test': meta_hog_test,
        'feature_type': 'HOG',
        'feature_size': X_hog.shape[1]
    }
    
    output_hog = output_path / 'detection_data_hog.pkl'
    with open(output_hog, 'wb') as f:
        pickle.dump(data_hog, f)
    print(f"   ✅ Datos HOG guardados en: {output_hog}")
    
    # Save processed data for BRISK
    data_brisk = {
        'X_train': X_brisk_train,
        'X_test': X_brisk_test,
        'y_train': y_brisk_train,
        'y_test': y_brisk_test,
        'metadata_train': meta_brisk_train,
        'metadata_test': meta_brisk_test,
        'feature_type': 'BRISK',
        'feature_size': X_brisk.shape[1]
    }
    
    output_brisk = output_path / 'detection_data_brisk.pkl'
    with open(output_brisk, 'wb') as f:
        pickle.dump(data_brisk, f)
    print(f"   ✅ Datos BRISK guardados en: {output_brisk}")
    print()
    
    # Print statistics
    print("📈 ESTADÍSTICAS DEL DATASET:")
    print()
    print("Coordenadas normalizadas (train):")
    print(f"   • X center: min={y_hog_train[:, 0].min():.3f}, max={y_hog_train[:, 0].max():.3f}, mean={y_hog_train[:, 0].mean():.3f}")
    print(f"   • Y center: min={y_hog_train[:, 1].min():.3f}, max={y_hog_train[:, 1].max():.3f}, mean={y_hog_train[:, 1].mean():.3f}")
    print(f"   • Width:    min={y_hog_train[:, 2].min():.3f}, max={y_hog_train[:, 2].max():.3f}, mean={y_hog_train[:, 2].mean():.3f}")
    print(f"   • Height:   min={y_hog_train[:, 3].min():.3f}, max={y_hog_train[:, 3].max():.3f}, mean={y_hog_train[:, 3].mean():.3f}")
    print()
    
    print("=" * 70)
    print("🎉 PREPARACIÓN COMPLETADA")
    print("=" * 70)
    
    return data_hog, data_brisk


if __name__ == '__main__':
    prepare_detection_dataset()
