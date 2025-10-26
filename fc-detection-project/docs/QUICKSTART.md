# Quick Start Guide

## Installation

```bash
# Clone repository
git clone <repo-url>
cd fc-detection-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Prepare Your Data

1. Organize your data:
```
data/
└── raw/
    ├── images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── annotations/
        ├── image1.xml
        ├── image2.xml
        └── ...
```

2. Prepare dataset:
```bash
# For HOG features
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type hog

# For BRISK features
python scripts/prepare_dataset.py \
    --images data/raw/images \
    --annotations data/raw/annotations \
    --feature-type brisk
```

## Train Models

### Train HOG Model
```bash
python scripts/train.py --config configs/hog_config.yaml
```

### Train BRISK Model
```bash
python scripts/train.py --config configs/brisk_config.yaml
```

### Monitor Training
```bash
# View TensorBoard
tensorboard --logdir logs/
```

## Evaluate Models

```bash
# Evaluate HOG model
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --feature-type hog

# Evaluate BRISK model
python scripts/evaluate.py \
    --model models/detection_brisk.h5 \
    --feature-type brisk
```

## Run Inference

```bash
# HOG inference
python scripts/inference.py \
    --model models/detection_hog.h5 \
    --image test.jpg \
    --feature-type hog \
    --show

# BRISK inference
python scripts/inference.py \
    --model models/detection_brisk.h5 \
    --image test.jpg \
    --feature-type brisk \
    --show
```

## Run Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Project Structure

```
fc-detection-project/
├── src/                    # Source code
│   ├── features/          # Feature extractors
│   ├── models/            # Neural network models
│   ├── data/              # Data utilities
│   ├── training/          # Training logic
│   └── evaluation/        # Metrics and evaluation
├── configs/               # Configuration files
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── notebooks/             # Jupyter notebooks
```

## Tips

### Customize Model Architecture

Edit `configs/hog_config.yaml` or `configs/brisk_config.yaml`:

```yaml
model:
  architecture:
    - units: 1024    # Increase for more capacity
      dropout: 0.4   # Adjust regularization
    - units: 512
      dropout: 0.3
    # ... add more layers
```

### Adjust Training Parameters

```yaml
training:
  epochs: 200          # Train longer
  batch_size: 64       # Larger batches
  learning_rate: 0.0001  # Lower learning rate
```

### Different Feature Settings

HOG:
```yaml
feature_extractor:
  params:
    orientations: 12    # More orientation bins
    pixels_per_cell: [16, 16]  # Coarser features
```

BRISK:
```yaml
feature_extractor:
  params:
    n_keypoints: 1024   # More keypoints
    threshold: 20       # Lower threshold (more keypoints)
```

## Troubleshooting

### Out of Memory
- Reduce batch_size
- Reduce model size (fewer units)
- Process fewer images at once

### Low Accuracy
- Increase model capacity
- Train for more epochs
- Adjust learning rate
- Try different feature parameters
- Ensure data quality

### Slow Training
- Increase batch_size (if memory allows)
- Reduce image resolution in feature extraction
- Use fewer keypoints (BRISK)
