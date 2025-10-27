#!/usr/bin/env python3
"""
Script para preparar el dataset para entrenamiento de detección.

Extrae características (HOG o BRISK) de las imágenes y las guarda con las etiquetas de bounding box.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import yaml

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.data.utils import parse_pascal_voc, normalize_bbox, DetectionDataset


def load_config(config_path: str) -> dict:
    """Carga configuración desde archivo YAML."""
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
    Preparar el dataset con características y etiquetas.
    
    Args:
        images_dir: Directorio con imágenes
        annotations_dir: Directorio con anotaciones XML
        feature_type: 'hog' o 'brisk'
        config: Diccionario de configuración
        output_path: Ruta para guardar el dataset procesado
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir)
    
    # Inicializar extractor de características
    feature_params = config['feature_extractor']['params']
    
    if feature_type == 'hog':
        extractor = HOGFeatureExtractor(**feature_params)
    elif feature_type == 'brisk':
        extractor = BRISKFeatureExtractor(**feature_params)
    else:
        raise ValueError(f"Tipo de características desconocido: {feature_type}")
    
    print(f"\n{'='*60}")
    print(f"Preparando dataset {feature_type.upper()}")
    print(f"{'='*60}")
    print(f"Extractor de características: {extractor}")
    print(f"Dimensión de características: {extractor.get_feature_dim()}")
    print(f"Directorio de imágenes: {images_dir}")
    print(f"Directorio de anotaciones: {annotations_dir}")
    print(f"{'='*60}\n")
    
    # Obtener todos los archivos XML
    xml_files = list(annotations_path.glob('*.xml'))
    
    if len(xml_files) == 0:
        print(f"ERROR: No se encontraron archivos XML en {annotations_dir}")
        return
    
    print(f"Se encontraron {len(xml_files)} archivos de anotaciones")
    
    features_list = []
    bboxes_list = []
    image_paths_list = []
    skipped = 0
    
    # Procesar cada anotación
    for xml_file in tqdm(xml_files, desc="Procesando imágenes"):
        try:
            # Parsear XML
            filename, bboxes = parse_pascal_voc(str(xml_file))
            
            # Saltar si no hay bboxes
            if len(bboxes) == 0:
                skipped += 1
                continue
            
            # Cargar imagen
            image_path = images_path / filename
            if not image_path.exists():
                # Intentar con extensión diferente
                image_path = images_path / (Path(filename).stem + '.jpg')
                if not image_path.exists():
                    image_path = images_path / (Path(filename).stem + '.png')
            
            if not image_path.exists():
                print(f"Advertencia: Imagen no encontrada: {filename}")
                skipped += 1
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Advertencia: Error al cargar imagen: {filename}")
                skipped += 1
                continue
            
            height, width = image.shape[:2]
            
            # Extraer características
            features = extractor.extract(image)
            
            # Procesar cada bbox (usar solo el primero para detección de un solo objeto)
            bbox = bboxes[0]
            normalized_bbox = normalize_bbox(bbox, width, height)
            
            # Agregar a listas
            features_list.append(features)
            bboxes_list.append(normalized_bbox)
            image_paths_list.append(str(image_path))
            
        except Exception as e:
            print(f"Error procesando {xml_file.name}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcesadas {len(features_list)} muestras")
    print(f"Omitidas {skipped} archivos")
    
    if len(features_list) == 0:
        print("ERROR: No se procesaron muestras")
        return
    
    # Convertir a arrays numpy
    features_array = np.array(features_list)
    bboxes_array = np.array(bboxes_list)
    
    print(f"\nForma de características: {features_array.shape}")
    print(f"Forma de bboxes: {bboxes_array.shape}")
    
    # Crear dataset
    dataset = DetectionDataset(
        features=features_array,
        bboxes=bboxes_array,
        image_paths=image_paths_list
    )
    
    # Guardar dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(str(output_file))
    
    print(f"\n✓ Dataset guardado en: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Preparar dataset de detección')
    parser.add_argument('--images', type=str, required=True, help='Directorio de imágenes')
    parser.add_argument('--annotations', type=str, required=True, help='Directorio de anotaciones')
    parser.add_argument('--feature-type', type=str, required=True, choices=['hog', 'brisk'], help='Tipo de características')
    parser.add_argument('--config', type=str, help='Ruta del archivo de configuración')
    parser.add_argument('--output', type=str, help='Ruta del archivo de salida')
    
    args = parser.parse_args()
    
    # Cargar configuración
    if args.config:
        config = load_config(args.config)
    else:
        # Usar configuración por defecto
        config_file = f"configs/{args.feature_type}_config.yaml"
        if Path(config_file).exists():
            config = load_config(config_file)
        else:
            print(f"ERROR: Archivo de configuración no encontrado: {config_file}")
            print("Por favor proporciona --config o crea el archivo de configuración por defecto")
            return
    
    # Establecer ruta de salida
    if args.output:
        output_path = args.output
    else:
        output_path = f"data/processed/detection_{args.feature_type}.pkl"
    
    # Preparar dataset
    prepare_dataset(
        images_dir=args.images,
        annotations_dir=args.annotations,
        feature_type=args.feature_type,
        config=config,
        output_path=output_path
    )


if __name__ == '__main__':
    main()
