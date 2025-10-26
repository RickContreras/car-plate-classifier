"""
Script para convertir el dataset de placas al formato YOLO.

Convierte las anotaciones XML (Pascal VOC) al formato YOLO:
- Crea estructura de carpetas train/val
- Genera archivos .txt con coordenadas normalizadas
- Crea archivo data.yaml para el entrenamiento
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import yaml


def parse_xml(xml_path):
    """Parse XML annotation file and extract bounding box."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Get bounding box
    obj = root.find('object')
    if obj is None:
        return None
        
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    
    return {
        'width': width,
        'height': height,
        'bbox': (xmin, ymin, xmax, ymax)
    }


def convert_to_yolo_format(bbox, img_width, img_height):
    """
    Convert Pascal VOC bbox to YOLO format.
    
    YOLO format: <class> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center and dimensions
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    bbox_width /= img_width
    bbox_height /= img_height
    
    # Class 0 for "plate" (only one class)
    return f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


def prepare_yolo_dataset(
    annotations_dir='data/raw/annotations',
    images_dir='data/raw/images',
    output_dir='data/yolo_dataset',
    train_split=0.8
):
    """
    Prepare YOLO dataset from XML annotations.
    
    Args:
        annotations_dir: Directory containing XML files
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        train_split: Fraction of data to use for training
    """
    print("🚀 Preparando dataset para YOLO...\n")
    
    # Create output directories
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images' / 'train'
    val_img_dir = output_path / 'images' / 'val'
    train_label_dir = output_path / 'labels' / 'train'
    val_label_dir = output_path / 'labels' / 'val'
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all XML files
    annotations_path = Path(annotations_dir)
    images_path = Path(images_dir)
    xml_files = list(annotations_path.glob('*.xml'))
    
    print(f"📁 Encontrados {len(xml_files)} archivos XML")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * train_split)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    print(f"📊 Train: {len(train_files)} | Validation: {len(val_files)}\n")
    
    # Process training files
    print("🔨 Procesando archivos de entrenamiento...")
    processed_train = process_files(
        train_files,
        images_path,
        train_img_dir,
        train_label_dir
    )
    
    # Process validation files
    print("🔨 Procesando archivos de validación...")
    processed_val = process_files(
        val_files,
        images_path,
        val_img_dir,
        val_label_dir
    )
    
    # Create data.yaml
    create_data_yaml(output_path, processed_train + processed_val)
    
    print("\n✅ Dataset YOLO preparado exitosamente!")
    print(f"📂 Ubicación: {output_dir}")
    print(f"✔️  Train: {processed_train} imágenes")
    print(f"✔️  Val: {processed_val} imágenes")


def process_files(xml_files, images_path, img_out_dir, label_out_dir):
    """Process a list of XML files and copy images/labels."""
    processed = 0
    
    for xml_file in xml_files:
        # Parse XML
        annotation = parse_xml(xml_file)
        if annotation is None:
            print(f"⚠️  Saltando {xml_file.name}: sin anotaciones")
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
            print(f"⚠️  Saltando {xml_file.name}: imagen no encontrada")
            continue
        
        # Convert to YOLO format
        yolo_label = convert_to_yolo_format(
            annotation['bbox'],
            annotation['width'],
            annotation['height']
        )
        
        # Copy image
        shutil.copy2(img_file, img_out_dir / img_file.name)
        
        # Save label
        label_file = label_out_dir / f"{img_name}.txt"
        with open(label_file, 'w') as f:
            f.write(yolo_label + '\n')
        
        processed += 1
    
    return processed


def create_data_yaml(output_path, total_images):
    """Create data.yaml file for YOLO training."""
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes
        'names': ['plate']  # Class names
    }
    
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n📝 Archivo data.yaml creado: {yaml_path}")


if __name__ == '__main__':
    prepare_yolo_dataset()
