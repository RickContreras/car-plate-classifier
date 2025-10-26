#!/usr/bin/env python3
"""
Script de ejemplo para usar el clasificador de placas sin interfaz gráfica.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import preprocess_image, load_config
from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor
from src.train_models import PlateClassifier


def classify_single_image(image_path: str, model_path: str, 
                         model_type: str = 'svm', feature_type: str = 'hog'):
    """
    Clasifica una imagen individual.
    
    Args:
        image_path: Ruta a la imagen
        model_path: Ruta al modelo entrenado
        model_type: Tipo de modelo ('svm', 'random_forest', 'neural_network')
        feature_type: Tipo de características ('hog', 'brisk')
    
    Returns:
        tuple: (predicción, confianza)
    """
    # Cargar configuración
    config = load_config()
    
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Preprocesar
    processed_image = preprocess_image(image, config)
    
    # Extraer características
    if feature_type == 'hog':
        extractor = HOGFeatureExtractor(config)
    elif feature_type == 'brisk':
        extractor = BRISKFeatureExtractor(config)
    else:
        raise ValueError(f"Tipo de característica desconocido: {feature_type}")
    
    features = extractor.extract(processed_image)
    features = features.reshape(1, -1)
    
    # Cargar modelo
    classifier = PlateClassifier(model_type=model_type)
    classifier.load(model_path)
    
    # Predecir
    prediction = classifier.predict(features)[0]
    confidence = classifier.predict_proba(features)[0]
    
    return prediction, confidence


def batch_classify(images_dir: str, model_path: str,
                   model_type: str = 'svm', feature_type: str = 'hog'):
    """
    Clasifica un lote de imágenes.
    
    Args:
        images_dir: Directorio con imágenes
        model_path: Ruta al modelo entrenado
        model_type: Tipo de modelo
        feature_type: Tipo de características
    """
    images_path = Path(images_dir)
    
    # Obtener todas las imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_path.glob(ext))
    
    if not image_files:
        print(f"No se encontraron imágenes en {images_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Clasificando {len(image_files)} imágenes...")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        try:
            prediction, confidence = classify_single_image(
                str(image_file), model_path, model_type, feature_type
            )
            
            label = "PLACA" if prediction == 1 else "NO PLACA"
            
            results.append({
                'file': image_file.name,
                'prediction': label,
                'confidence': confidence
            })
            
            print(f"{i}. {image_file.name:40s} -> {label:10s} "
                  f"(Confianza: {confidence*100:.2f}%)")
            
        except Exception as e:
            print(f"{i}. {image_file.name:40s} -> ERROR: {e}")
    
    # Resumen
    placas = sum(1 for r in results if r['prediction'] == 'PLACA')
    no_placas = len(results) - placas
    
    print(f"\n{'='*60}")
    print("RESUMEN:")
    print(f"   Total procesadas: {len(results)}")
    print(f"   Placas detectadas: {placas}")
    print(f"   No placas: {no_placas}")
    print(f"{'='*60}")


def main():
    """Función principal de ejemplo."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clasificador de placas vehiculares'
    )
    parser.add_argument('--image', type=str, help='Ruta a una imagen individual')
    parser.add_argument('--dir', type=str, help='Directorio con imágenes')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al modelo entrenado')
    parser.add_argument('--model-type', type=str, default='svm',
                       choices=['svm', 'random_forest', 'neural_network'],
                       help='Tipo de modelo')
    parser.add_argument('--features', type=str, default='hog',
                       choices=['hog', 'brisk'],
                       help='Tipo de características')
    
    args = parser.parse_args()
    
    if args.image:
        # Clasificar imagen individual
        try:
            prediction, confidence = classify_single_image(
                args.image, args.model, args.model_type, args.features
            )
            
            label = "PLACA DETECTADA" if prediction == 1 else "NO ES UNA PLACA"
            
            print(f"\n{'='*60}")
            print(f"Resultado: {label}")
            print(f"Confianza: {confidence*100:.2f}%")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.dir:
        # Clasificar lote de imágenes
        try:
            batch_classify(args.dir, args.model, args.model_type, args.features)
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  # Clasificar una imagen:")
        print("  python example_usage.py --image imagen.jpg --model models/svm_hog.pkl")
        print("\n  # Clasificar un directorio:")
        print("  python example_usage.py --dir test_images/ --model models/neural_network_hog.h5 "
              "--model-type neural_network")


if __name__ == "__main__":
    main()
