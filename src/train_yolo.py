"""
Script para entrenar YOLOv8 para detección de placas vehiculares.

Entrena un modelo YOLOv8 desde cero usando el dataset preparado.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path


def train_yolo_model(
    data_yaml='data/yolo_dataset/data.yaml',
    model_size='n',  # n, s, m, l, x (nano, small, medium, large, extra)
    epochs=50,
    imgsz=640,
    batch=16,
    project='models/yolo',
    name='plate_detector'
):
    """
    Train YOLOv8 model for plate detection.
    
    Args:
        data_yaml: Path to data.yaml file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        project: Project directory to save results
        name: Experiment name
    """
    print("=" * 70)
    print("🚀 ENTRENAMIENTO DE YOLOv8 PARA DETECCIÓN DE PLACAS")
    print("=" * 70)
    print()
    
    # Verify data.yaml exists
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"❌ No se encontró {data_yaml}")
    
    # Load and display dataset info
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("📊 CONFIGURACIÓN DEL DATASET:")
    print(f"   • Ruta: {data_config['path']}")
    print(f"   • Clases: {data_config['nc']} ({data_config['names']})")
    print(f"   • Train: {data_config['train']}")
    print(f"   • Val: {data_config['val']}")
    print()
    
    print("🏗️  CONFIGURACIÓN DEL MODELO:")
    print(f"   • Modelo: YOLOv8{model_size}")
    print(f"   • Épocas: {epochs}")
    print(f"   • Tamaño de imagen: {imgsz}x{imgsz}")
    print(f"   • Batch size: {batch}")
    print(f"   • Carpeta de salida: {project}/{name}")
    print()
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    print("🎯 INICIANDO ENTRENAMIENTO...")
    print("-" * 70)
    print()
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        patience=10,  # Early stopping patience
        save=True,
        device='cpu',  # Change to '0' for GPU or 'cpu' for CPU
        workers=4,
        verbose=True,
        plots=True  # Generate training plots
    )
    
    print()
    print("=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print()
    
    # Display results
    print("📈 RESULTADOS DEL ENTRENAMIENTO:")
    model_dir = Path(project) / name
    print(f"   • Modelo entrenado: {model_dir}/weights/best.pt")
    print(f"   • Último modelo: {model_dir}/weights/last.pt")
    print(f"   • Gráficas: {model_dir}/")
    print()
    
    # Validate the model
    print("🔍 VALIDANDO MODELO...")
    print("-" * 70)
    metrics = model.val()
    
    print()
    print("📊 MÉTRICAS DE VALIDACIÓN:")
    print(f"   • mAP50: {metrics.box.map50:.4f}")
    print(f"   • mAP50-95: {metrics.box.map:.4f}")
    print(f"   • Precision: {metrics.box.mp:.4f}")
    print(f"   • Recall: {metrics.box.mr:.4f}")
    print()
    
    print("🎉 ¡ENTRENAMIENTO EXITOSO!")
    print(f"💾 Modelo guardado en: {model_dir}/weights/best.pt")
    print()
    
    return model, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 for plate detection')
    parser.add_argument('--model', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=extra)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    
    args = parser.parse_args()
    
    # Train model
    train_yolo_model(
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
