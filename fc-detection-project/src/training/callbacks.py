"""
Custom callbacks for training.
"""

from tensorflow import keras
from pathlib import Path
from typing import Optional, List


def get_callbacks(
    model_name: str,
    save_dir: str = "models",
    patience: int = 15,
    reduce_lr_patience: int = 7,
    min_lr: float = 1e-7,
    monitor: str = 'val_loss',
    mode: str = 'min'
) -> List[keras.callbacks.Callback]:
    """
    Get standard callbacks for training.
    
    Args:
        model_name: Name of the model
        save_dir: Directory to save checkpoints
        patience: Early stopping patience
        reduce_lr_patience: ReduceLROnPlateau patience
        min_lr: Minimum learning rate
        monitor: Metric to monitor
        mode: 'min' or 'max'
        
    Returns:
        List of callbacks
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    callbacks = []
    
    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        mode=mode,
        verbose=1
    )
    callbacks.append(early_stop)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        mode=mode,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model checkpoint
    checkpoint_path = save_path / f"{model_name}_best.h5"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # TensorBoard
    log_dir = Path("logs") / model_name
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        histogram_freq=0,
        write_graph=True,
        update_freq='epoch'
    )
    callbacks.append(tensorboard)
    
    return callbacks


class IoUCallback(keras.callbacks.Callback):
    """Custom callback to calculate IoU during training."""
    
    def __init__(self, validation_data, name='avg_iou'):
        """
        Initialize IoU callback.
        
        Args:
            validation_data: Tuple of (X_val, y_val)
            name: Name for the metric
        """
        super().__init__()
        self.validation_data = validation_data
        self.name = name
    
    def on_epoch_end(self, epoch, logs=None):
        """Calculate IoU at end of epoch."""
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Calculate IoU
        iou_scores = []
        for pred, true in zip(y_pred, y_val):
            iou = self._calculate_iou(pred, true)
            iou_scores.append(iou)
        
        avg_iou = sum(iou_scores) / len(iou_scores)
        
        # Add to logs
        logs[self.name] = avg_iou
        
        print(f" - {self.name}: {avg_iou:.4f}")
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """Calculate IoU between two normalized boxes."""
        # Convert from center format to corner format
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
        
        # Calculate intersection
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
