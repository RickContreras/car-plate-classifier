#!/usr/bin/env python3
"""
Script para realizar inferencia con modelos de detección entrenados.
"""

import argparse
import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import os

# Desactivar GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras
from src.features import HOGFeatureExtractor, BRISKFeatureExtractor
from src.data.utils import denormalize_bbox


def load_config(config_path: str) -> dict:
    """Cargar configuración desde archivo YAML."""
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
    Realizar inferencia en una sola imagen.
    
    Args:
        model_path: Ruta al modelo entrenado
        image_path: Ruta a la imagen de entrada
        feature_type: 'hog' o 'brisk'
        config: Diccionario de configuración
        output_path: Ruta para guardar imagen resultado
        show: Si se debe mostrar el resultado
    """
    print(f"\n{'='*60}")
    print(f"Inferencia con Modelo {feature_type.upper()}")
    print(f"{'='*60}\n")
    
    # Cargar modelo
    print(f"Cargando modelo: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("  Modelo cargado")
    
    # Inicializar extractor de características
    feature_params = config['feature_extractor']['params']
    
    if feature_type == 'hog':
        extractor = HOGFeatureExtractor(**feature_params)
    elif feature_type == 'brisk':
        extractor = BRISKFeatureExtractor(**feature_params)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    print(f"  Extractor de características inicializado: {extractor}")
    
    # Cargar imagen
    print(f"\nCargando imagen: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"ERROR: No se pudo cargar la imagen: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"  Imagen cargada: {width}x{height}")
    
    # Extraer características
    print("\nExtrayendo características...")
    features = extractor.extract(image)
    features = np.expand_dims(features, axis=0)  # Agregar dimensión de batch
    print(f"  Características extraídas: {features.shape}")
    
    # Predecir bounding box
    print("\nPrediciendo bounding box...")
    pred_bbox_norm = model.predict(features, verbose=0)[0]
    print(f"  Bbox normalizado: [{pred_bbox_norm[0]:.3f}, {pred_bbox_norm[1]:.3f}, {pred_bbox_norm[2]:.3f}, {pred_bbox_norm[3]:.3f}]")
    
    # Desnormalizar bbox
    pred_bbox = denormalize_bbox(pred_bbox_norm, width, height)
    xmin, ymin, xmax, ymax = pred_bbox
    print(f"  Bbox en píxeles: ({xmin}, {ymin}, {xmax}, {ymax})")
    
    # Dibujar bounding box
    result_image = image.copy()
    cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    # Agregar etiqueta
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
    
    # Guardar resultado
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"\n  Resultado guardado en: {output_path}")
    
    # Mostrar resultado
    if show:
        cv2.imshow('Resultado de Detección', result_image)
        print("\nPresiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n{'='*60}")
    print("¡Inferencia completada!")
    print(f"{'='*60}\n")
    
    return result_image, pred_bbox


def main():
    parser = argparse.ArgumentParser(description='Realizar inferencia con modelo de detección')
    parser.add_argument('--model', type=str, required=True, help='Ruta del archivo de modelo')
    parser.add_argument('--image', type=str, required=True, help='Ruta de la imagen de entrada')
    parser.add_argument('--feature-type', type=str, required=True, choices=['hog', 'brisk'], help='Tipo de características')
    parser.add_argument('--config', type=str, help='Ruta del archivo de configuración')
    parser.add_argument('--output', type=str, help='Ruta de la imagen de salida')
    parser.add_argument('--show', action='store_true', help='Mostrar resultado')
    
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
