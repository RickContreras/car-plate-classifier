#!/usr/bin/env python3
"""
Script to train detection models.

Trains a fully connected neural network for bounding box regression.
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable GPU to avoid CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import load_dataset
from src.models import FCNetwork
from src.training import Trainer, get_callbacks
from src.training.callbacks import IoUCallback
from src.evaluation import MetricsCalculator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_training_history(history, save_path: str):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Training and Validation MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training history plot saved to: {save_path}")
    plt.close()


def train_model(config_path: str, dataset_path: str, feature_type: str):
    """
    Train detection model.
    
    Args:
        config_path: Path to config file
        dataset_path: Path to dataset file
        feature_type: 'hog' or 'brisk'
    """
    # Load config
    config = load_config(config_path)
    
    print(f"\n{'='*60}")
    print(f"Training {feature_type.upper()} Detection Model")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Split dataset
    data_config = config['data']
    train_dataset, val_dataset = dataset.split(
        train_ratio=data_config['train_ratio'],
        shuffle=data_config['shuffle'],
        seed=data_config.get('seed', 42)
    )
    
    X_train, y_train = train_dataset.features, train_dataset.bboxes
    X_val, y_val = val_dataset.features, val_dataset.bboxes
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Feature dimension: {X_train.shape[1]}")
    
    # Create model
    model_config = config['model']
    architecture = [layer['units'] for layer in model_config['architecture']]
    activations = [layer['activation'] for layer in model_config['architecture']]
    dropout_rates = [layer.get('dropout', 0.0) for layer in model_config['architecture']]
    
    use_batch_norm = any(layer.get('batch_norm', False) for layer in model_config['architecture'][:-1])
    
    print("\nBuilding model...")
    fc_model = FCNetwork(
        input_dim=X_train.shape[1],
        architecture=architecture,
        activations=activations,
        use_batch_norm=use_batch_norm,
        dropout_rates=dropout_rates,
        l2_reg=model_config.get('l2_reg', 0.0),
        name=model_config['name']
    )
    
    # Compile model
    training_config = config['training']
    fc_model.compile(
        optimizer=training_config['optimizer'],
        learning_rate=training_config['learning_rate'],
        loss=training_config['loss'],
        metrics=training_config['metrics']
    )
    
    print("\nModel Summary:")
    fc_model.summary()
    
    # Setup callbacks
    paths = config['paths']
    callbacks = get_callbacks(
        model_name=model_config['name'],
        save_dir=paths['models'],
        patience=training_config['callbacks']['early_stopping']['patience'],
        reduce_lr_patience=training_config['callbacks']['reduce_lr']['patience'],
        min_lr=training_config['callbacks']['reduce_lr']['min_lr'],
        monitor=training_config['callbacks']['early_stopping']['monitor'],
        mode=training_config['callbacks']['early_stopping']['mode']
    )
    
    # Add IoU callback
    iou_callback = IoUCallback(validation_data=(X_val, y_val))
    callbacks.append(iou_callback)
    
    # Create trainer
    trainer = Trainer(
        model=fc_model.get_model(),
        save_dir=paths['models'],
        log_dir=paths['logs'],
        name=model_config['name']
    )
    
    # Train model
    print("\nStarting training...\n")
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model
    model_path = Path(paths['models']) / f"{model_config['name']}.h5"
    trainer.save_model(str(model_path))
    
    # Save history
    trainer.save_history()
    
    # Plot training history
    results_dir = Path(paths['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / f"{model_config['name']}_training.png"
    plot_training_history(history, str(plot_path))
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_pred = trainer.predict(X_val)
    
    metrics_calc = MetricsCalculator(
        iou_thresholds=config['evaluation']['iou_thresholds']
    )
    metrics_calc.update(y_val, y_pred)
    metrics = metrics_calc.compute()
    metrics_calc.print_metrics(metrics)
    
    # Save metrics
    import json
    metrics_path = results_dir / f"{model_config['name']}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train detection model')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--dataset', type=str, help='Dataset file path')
    parser.add_argument('--feature-type', type=str, choices=['hog', 'brisk'], help='Feature type')
    
    args = parser.parse_args()
    
    # Determine feature type from config or argument
    if args.feature_type:
        feature_type = args.feature_type
    else:
        # Extract from config filename
        config_name = Path(args.config).stem
        if 'hog' in config_name:
            feature_type = 'hog'
        elif 'brisk' in config_name:
            feature_type = 'brisk'
        else:
            print("ERROR: Cannot determine feature type. Use --feature-type")
            return
    
    # Set dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = f"data/processed/detection_{feature_type}.pkl"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print("Run prepare_dataset.py first")
        return
    
    # Train model
    train_model(args.config, dataset_path, feature_type)


if __name__ == '__main__':
    main()
