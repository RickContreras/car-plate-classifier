# API Documentation

## Feature Extractors

### HOGFeatureExtractor

Extracts Histogram of Oriented Gradients features.

```python
from src.features import HOGFeatureExtractor

extractor = HOGFeatureExtractor(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(3, 3),
    target_size=(200, 200)
)

features = extractor.extract(image)
```

**Parameters:**
- `orientations` (int): Number of orientation bins (default: 9)
- `pixels_per_cell` (tuple): Size of a cell in pixels (default: (8, 8))
- `cells_per_block` (tuple): Number of cells in each block (default: (3, 3))
- `block_norm` (str): Normalization method (default: 'L2-Hys')
- `target_size` (tuple): Image resize target (default: (200, 200))
- `transform_sqrt` (bool): Apply power law compression (default: True)

**Methods:**
- `extract(image)`: Extract features from single image
- `extract_batch(images)`: Extract features from multiple images
- `get_feature_dim()`: Get feature vector dimension
- `visualize_hog(image)`: Visualize HOG features

### BRISKFeatureExtractor

Extracts Binary Robust Invariant Scalable Keypoints features.

```python
from src.features import BRISKFeatureExtractor

extractor = BRISKFeatureExtractor(
    n_keypoints=512,
    target_size=(200, 200),
    threshold=30
)

features = extractor.extract(image)
```

**Parameters:**
- `n_keypoints` (int): Number of keypoints to extract (default: 512)
- `target_size` (tuple): Image resize target (default: (200, 200))
- `threshold` (int): AGAST detection threshold (default: 30)
- `octaves` (int): Detection octaves (default: 3)
- `pattern_scale` (float): Pattern sampling scale (default: 1.0)

**Methods:**
- `extract(image)`: Extract features from single image
- `extract_batch(images)`: Extract features from multiple images
- `get_feature_dim()`: Get feature vector dimension
- `visualize_keypoints(image)`: Visualize detected keypoints

## Models

### FCNetwork

Fully Connected Neural Network for bbox regression.

```python
from src.models import FCNetwork

model = FCNetwork(
    input_dim=8100,
    architecture=[512, 256, 128, 64, 4],
    use_batch_norm=True
)

model.compile(learning_rate=0.001)
```

**Parameters:**
- `input_dim` (int): Input feature dimension
- `architecture` (list): List of layer sizes
- `activations` (list): Activation per layer (default: relu + sigmoid)
- `use_batch_norm` (bool): Use batch normalization (default: True)
- `dropout_rates` (list): Dropout rate per layer
- `l2_reg` (float): L2 regularization factor (default: 0.0)

**Methods:**
- `compile(optimizer, learning_rate, loss, metrics)`: Compile model
- `get_model()`: Get Keras model
- `save(filepath)`: Save model
- `load(filepath)`: Load model (static method)

## Data

### DetectionDataset

Dataset class for detection tasks.

```python
from src.data import DetectionDataset

dataset = DetectionDataset(
    features=features_array,
    bboxes=bboxes_array,
    image_paths=paths_list
)

train_ds, val_ds = dataset.split(train_ratio=0.8)
```

**Methods:**
- `__len__()`: Get dataset size
- `__getitem__(idx)`: Get single sample
- `split(train_ratio, shuffle, seed)`: Split into train/val
- `save(filepath)`: Save to pickle file
- `load(filepath)`: Load from pickle file (static method)

### Utilities

```python
from src.data import normalize_bbox, denormalize_bbox

# Normalize: pixels -> [0, 1]
norm_bbox = normalize_bbox((xmin, ymin, xmax, ymax), img_w, img_h)

# Denormalize: [0, 1] -> pixels
bbox = denormalize_bbox((x_c, y_c, w, h), img_w, img_h)
```

## Training

### Trainer

Training manager for models.

```python
from src.training import Trainer

trainer = Trainer(
    model=keras_model,
    save_dir='models',
    name='detection_hog'
)

history = trainer.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    callbacks=callbacks
)
```

**Methods:**
- `train(X_train, y_train, X_val, y_val, ...)`: Train model
- `evaluate(X_test, y_test)`: Evaluate model
- `predict(X)`: Make predictions
- `save_model(filepath)`: Save trained model
- `save_history(filepath)`: Save training history

### Callbacks

```python
from src.training import get_callbacks

callbacks = get_callbacks(
    model_name='detection_hog',
    patience=15,
    reduce_lr_patience=7
)
```

Returns list with:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- TensorBoard

## Evaluation

### MetricsCalculator

Calculate detection metrics.

```python
from src.evaluation import MetricsCalculator

calc = MetricsCalculator(iou_thresholds=[0.5, 0.75])
calc.update(y_true, y_pred)
metrics = calc.compute()
calc.print_metrics()
```

**Metrics:**
- `avg_iou`: Average IoU
- `median_iou`: Median IoU
- `mae`: Mean Absolute Error
- `accuracy@threshold`: Accuracy at IoU threshold

### Functions

```python
from src.evaluation import calculate_iou, evaluate_detections

# Single IoU
iou = calculate_iou(pred_bbox, true_bbox)

# Full evaluation
metrics = evaluate_detections(y_true, y_pred, iou_threshold=0.5)
```

## Scripts Usage

### Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog \
    --output data/processed/detection_hog.pkl
```

### Train Model

```bash
python scripts/train.py \
    --config configs/hog_config.yaml \
    --dataset data/processed/detection_hog.pkl
```

### Evaluate Model

```bash
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --dataset data/processed/detection_hog.pkl \
    --feature-type hog
```

### Inference

```bash
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --image test_image.jpg \
    --feature-type hog \
    --output result.jpg \
    --show
```
