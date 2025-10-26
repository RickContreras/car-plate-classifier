"""
Script para entrenar Red Neuronal Completamente Conectada para detección de placas.

Entrena modelos de regresión para predecir coordenadas de bounding boxes.
"""

import os
# Force CPU usage to avoid CUDA/ptxas issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt


def create_detection_model(input_dim, model_name='FCN'):
    """
    Create fully connected neural network for bounding box regression.
    
    Args:
        input_dim: Number of input features
        model_name: Name for the model
        
    Returns:
        keras.Model: Compiled model
    """
    model = models.Sequential(name=model_name)
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Dense layers with dropout and batch normalization
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    # Output layer: 4 values (x_center, y_center, width, height) normalized 0-1
    model.add(layers.Dense(4, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error
    )
    
    return model


def plot_training_history(history, feature_type, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title(f'{feature_type} - Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title(f'{feature_type} - Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_history_{feature_type.lower()}.png', dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test, metadata_test, feature_type):
    """Evaluate model and print metrics."""
    print(f"\n📊 EVALUACIÓN DEL MODELO ({feature_type}):")
    print("=" * 70)
    
    # Get predictions
    y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    # Calculate per-coordinate MAE
    mae_x = np.mean(np.abs(y_test[:, 0] - y_pred[:, 0]))
    mae_y = np.mean(np.abs(y_test[:, 1] - y_pred[:, 1]))
    mae_w = np.mean(np.abs(y_test[:, 2] - y_pred[:, 2]))
    mae_h = np.mean(np.abs(y_test[:, 3] - y_pred[:, 3]))
    
    print(f"\n🎯 Métricas Generales:")
    print(f"   • MSE (Mean Squared Error): {mse:.6f}")
    print(f"   • MAE (Mean Absolute Error): {mae:.6f}")
    print()
    
    print(f"📍 MAE por Coordenada (normalized 0-1):")
    print(f"   • X center: {mae_x:.6f} ({mae_x*100:.2f}% de error)")
    print(f"   • Y center: {mae_y:.6f} ({mae_y*100:.2f}% de error)")
    print(f"   • Width:    {mae_w:.6f} ({mae_w*100:.2f}% de error)")
    print(f"   • Height:   {mae_h:.6f} ({mae_h*100:.2f}% de error)")
    print()
    
    # Calculate IoU (Intersection over Union) for evaluation
    ious = []
    for i in range(len(y_test)):
        # Convert normalized coordinates to absolute (assuming some standard size)
        # For actual evaluation, we'd use the real image dimensions from metadata
        true_box = y_test[i]
        pred_box = y_pred[i]
        
        # Calculate IoU (simplified version with normalized coordinates)
        iou = calculate_iou_normalized(true_box, pred_box)
        ious.append(iou)
    
    avg_iou = np.mean(ious)
    print(f"📐 IoU (Intersection over Union):")
    print(f"   • Average IoU: {avg_iou:.4f}")
    print(f"   • IoU > 0.5: {np.sum(np.array(ious) > 0.5)}/{len(ious)} ({np.sum(np.array(ious) > 0.5)/len(ious)*100:.1f}%)")
    print(f"   • IoU > 0.7: {np.sum(np.array(ious) > 0.7)}/{len(ious)} ({np.sum(np.array(ious) > 0.7)/len(ious)*100:.1f}%)")
    print()
    
    return {
        'mse': mse,
        'mae': mae,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'mae_w': mae_w,
        'mae_h': mae_h,
        'avg_iou': avg_iou
    }


def calculate_iou_normalized(box1, box2):
    """Calculate IoU for normalized bounding boxes (x_center, y_center, width, height)."""
    # Convert from center format to corner format
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2
    
    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2
    
    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0
    
    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    
    # Calculate union
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def train_detection_model(
    data_path,
    feature_type='HOG',
    epochs=100,
    batch_size=32,
    output_dir='models/detection_nn'
):
    """
    Train fully connected neural network for plate detection.
    
    Args:
        data_path: Path to prepared data pickle file
        feature_type: Type of features ('HOG' or 'BRISK')
        epochs: Number of training epochs
        batch_size: Batch size for training
        output_dir: Directory to save models and plots
    """
    print("=" * 70)
    print(f"🚀 ENTRENAMIENTO DE RED NEURONAL PARA DETECCIÓN - {feature_type}")
    print("=" * 70)
    print()
    
    # Load data
    print(f"📂 Cargando datos desde: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    metadata_test = data['metadata_test']
    
    print(f"   ✅ Datos cargados")
    print(f"   • Train: {len(X_train)} muestras")
    print(f"   • Test: {len(X_test)} muestras")
    print(f"   • Features: {X_train.shape[1]} dimensiones")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("🏗️  Creando arquitectura de red neuronal...")
    model = create_detection_model(
        input_dim=X_train.shape[1],
        model_name=f'DetectionNN_{feature_type}'
    )
    
    print(f"\n📋 Resumen del Modelo:")
    model.summary()
    print()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("🎯 Iniciando entrenamiento...")
    print("-" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print()
    print("=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, metadata_test, feature_type)
    
    # Save model
    model_filename = f'detection_nn_{feature_type.lower()}.h5'
    model_path = output_path / model_filename
    model.save(model_path)
    print(f"💾 Modelo guardado en: {model_path}")
    
    # Plot training history
    plot_training_history(history, feature_type, output_path)
    print(f"📊 Gráficas guardadas en: {output_path}")
    
    print()
    print("=" * 70)
    print("🎉 PROCESO COMPLETADO")
    print("=" * 70)
    
    return model, history, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Detection Neural Network')
    parser.add_argument('--features', type=str, default='both',
                        choices=['hog', 'brisk', 'both'],
                        help='Type of features to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Train HOG model
    if args.features in ['hog', 'both']:
        print("\n" + "🔵" * 35)
        print("ENTRENANDO MODELO CON CARACTERÍSTICAS HOG")
        print("🔵" * 35 + "\n")
        
        train_detection_model(
            data_path='data/detection_dataset/detection_data_hog.pkl',
            feature_type='HOG',
            epochs=args.epochs,
            batch_size=args.batch
        )
    
    # Train BRISK model
    if args.features in ['brisk', 'both']:
        print("\n" + "🟢" * 35)
        print("ENTRENANDO MODELO CON CARACTERÍSTICAS BRISK")
        print("🟢" * 35 + "\n")
        
        train_detection_model(
            data_path='data/detection_dataset/detection_data_brisk.pkl',
            feature_type='BRISK',
            epochs=args.epochs,
            batch_size=args.batch
        )
