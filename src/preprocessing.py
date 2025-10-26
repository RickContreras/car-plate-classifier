"""
Módulo de preprocesamiento de imágenes para el proyecto de detección de placas.
Incluye funciones para cargar datos, parsear anotaciones XML y preprocesar imágenes.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, List, Dict
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Carga la configuración desde el archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        dict: Diccionario con la configuración
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_xml_annotation(xml_path: str) -> Dict:
    """
    Parsea un archivo XML de anotación y extrae información de la imagen y bounding box.
    
    Args:
        xml_path: Ruta al archivo XML
        
    Returns:
        dict: Diccionario con filename, width, height, y bounding box (xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotation = {
        'filename': root.find('filename').text,
        'width': int(root.find('size/width').text),
        'height': int(root.find('size/height').text),
        'bboxes': []
    }
    
    for obj in root.findall('object'):
        bbox = {
            'name': obj.find('name').text,
            'xmin': int(obj.find('bndbox/xmin').text),
            'ymin': int(obj.find('bndbox/ymin').text),
            'xmax': int(obj.find('bndbox/xmax').text),
            'ymax': int(obj.find('bndbox/ymax').text)
        }
        annotation['bboxes'].append(bbox)
    
    return annotation


def crop_plate_region(image: np.ndarray, bbox: Dict) -> np.ndarray:
    """
    Recorta la región de la placa desde la imagen usando el bounding box.
    
    Args:
        image: Imagen original
        bbox: Diccionario con coordenadas del bounding box
        
    Returns:
        np.ndarray: Imagen recortada de la placa
    """
    return image[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]


def preprocess_image(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Preprocesa una imagen según la configuración especificada.
    
    Args:
        image: Imagen a preprocesar
        config: Configuración de preprocesamiento
        
    Returns:
        np.ndarray: Imagen preprocesada
    """
    processed = image.copy()
    
    # Resize
    if config['preprocessing']['resize']:
        img_size = tuple(config['data']['img_size'])
        processed = cv2.resize(processed, img_size)
    
    # Convert to grayscale
    if config['preprocessing']['grayscale']:
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization
    if config['preprocessing']['equalize_hist']:
        processed = cv2.equalizeHist(processed)
    
    # Normalize
    if config['preprocessing']['normalize']:
        processed = processed.astype(np.float32) / 255.0
    
    return processed


def load_dataset(config: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Carga el dataset completo de imágenes de placas.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Tuple: (imágenes, etiquetas, nombres de archivos)
    """
    raw_path = Path(config['data']['raw_path'])
    images_path = raw_path / 'images'
    annotations_path = raw_path / 'annotations'
    
    images_list = []
    labels_list = []
    filenames = []
    
    # Obtener lista de archivos XML
    xml_files = sorted(annotations_path.glob('*.xml'))
    
    print(f"📊 Procesando {len(xml_files)} imágenes...")
    
    for xml_file in xml_files:
        # Parsear anotación
        annotation = parse_xml_annotation(str(xml_file))
        image_path = images_path / annotation['filename']
        
        # Verificar que la imagen existe
        if not image_path.exists():
            print(f"⚠️  Imagen no encontrada: {image_path}")
            continue
        
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️  Error al cargar: {image_path}")
            continue
        
        # Procesar cada bounding box (placa)
        for bbox in annotation['bboxes']:
            # Recortar región de la placa
            plate_image = crop_plate_region(image, bbox)
            
            # Preprocesar
            processed_image = preprocess_image(plate_image, config)
            
            images_list.append(processed_image)
            labels_list.append(1)  # 1 = placa
            filenames.append(annotation['filename'])
    
    print(f"✅ {len(images_list)} placas procesadas correctamente")
    
    return np.array(images_list), np.array(labels_list), filenames


def create_negative_samples(config: dict, num_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea muestras negativas (regiones sin placas) para balancear el dataset.
    
    Args:
        config: Configuración del proyecto
        num_samples: Número de muestras negativas a generar
        
    Returns:
        Tuple: (imágenes negativas, etiquetas)
    """
    raw_path = Path(config['data']['raw_path'])
    images_path = raw_path / 'images'
    annotations_path = raw_path / 'annotations'
    
    negative_images = []
    negative_labels = []
    
    xml_files = sorted(annotations_path.glob('*.xml'))
    
    if num_samples is None:
        num_samples = len(xml_files)
    
    print(f"📊 Generando {num_samples} muestras negativas...")
    
    samples_per_image = max(1, num_samples // len(xml_files))
    
    for xml_file in xml_files[:num_samples]:
        annotation = parse_xml_annotation(str(xml_file))
        image_path = images_path / annotation['filename']
        
        if not image_path.exists():
            continue
        
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        img_size = config['data']['img_size']
        
        # Generar regiones aleatorias que NO sean placas
        for _ in range(samples_per_image):
            # Intentar varias veces hasta encontrar una región sin solapamiento
            for attempt in range(10):
                x = np.random.randint(0, max(1, w - img_size[0]))
                y = np.random.randint(0, max(1, h - img_size[1]))
                
                # Verificar que no solape con ninguna placa
                bbox_candidate = {
                    'xmin': x,
                    'ymin': y,
                    'xmax': x + img_size[0],
                    'ymax': y + img_size[1]
                }
                
                overlaps = False
                for bbox in annotation['bboxes']:
                    if bboxes_overlap(bbox_candidate, bbox):
                        overlaps = True
                        break
                
                if not overlaps:
                    negative_region = image[y:y+img_size[1], x:x+img_size[0]]
                    processed = preprocess_image(negative_region, config)
                    negative_images.append(processed)
                    negative_labels.append(0)  # 0 = no placa
                    break
        
        if len(negative_images) >= num_samples:
            break
    
    print(f"✅ {len(negative_images)} muestras negativas generadas")
    
    return np.array(negative_images), np.array(negative_labels)


def bboxes_overlap(bbox1: Dict, bbox2: Dict, threshold: float = 0.3) -> bool:
    """
    Verifica si dos bounding boxes se solapan.
    
    Args:
        bbox1: Primer bounding box
        bbox2: Segundo bounding box
        threshold: Umbral de IoU para considerar solapamiento
        
    Returns:
        bool: True si hay solapamiento significativo
    """
    # Calcular área de intersección
    x_left = max(bbox1['xmin'], bbox2['xmin'])
    y_top = max(bbox1['ymin'], bbox2['ymin'])
    x_right = min(bbox1['xmax'], bbox2['xmax'])
    y_bottom = min(bbox1['ymax'], bbox2['ymax'])
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calcular áreas de cada bbox
    bbox1_area = (bbox1['xmax'] - bbox1['xmin']) * (bbox1['ymax'] - bbox1['ymin'])
    bbox2_area = (bbox2['xmax'] - bbox2['xmin']) * (bbox2['ymax'] - bbox2['ymin'])
    
    # Calcular IoU
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    
    return iou > threshold


def prepare_dataset(config: dict) -> Tuple:
    """
    Prepara el dataset completo con muestras positivas y negativas.
    
    Args:
        config: Configuración del proyecto
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n" + "="*60)
    print("🔄 PREPARANDO DATASET")
    print("="*60)
    
    # Cargar muestras positivas (placas)
    positive_images, positive_labels, _ = load_dataset(config)
    
    # Generar muestras negativas
    negative_images, negative_labels = create_negative_samples(
        config, 
        num_samples=len(positive_images)
    )
    
    # Combinar datasets
    X = np.concatenate([positive_images, negative_images])
    y = np.concatenate([positive_labels, negative_labels])
    
    print(f"\n📊 Dataset completo:")
    print(f"   • Placas (positivos): {len(positive_images)}")
    print(f"   • No placas (negativos): {len(negative_images)}")
    print(f"   • Total: {len(X)}")
    
    # Split dataset
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n📊 Split del dataset:")
    print(f"   • Entrenamiento: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"   • Prueba: {len(X_test)} ({test_size*100:.0f}%)")
    print("="*60)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Prueba del módulo
    config = load_config()
    X_train, X_test, y_train, y_test = prepare_dataset(config)
    print(f"\n✅ Preprocesamiento completado exitosamente")
