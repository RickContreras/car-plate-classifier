#!/bin/bash
# Complete pipeline example for FC Detection Project

echo "=================================="
echo "FC Detection - Complete Pipeline"
echo "=================================="

# Configuration
IMAGES_DIR="data/raw/images"
ANNOTATIONS_DIR="data/raw/annotations"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Prepare HOG dataset
echo -e "\n${BLUE}Step 1: Preparing HOG dataset...${NC}"
python scripts/prepare_dataset.py \
    --images $IMAGES_DIR \
    --annotations $ANNOTATIONS_DIR \
    --feature-type hog \
    --output data/processed/detection_hog.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ HOG dataset prepared${NC}"
else
    echo "Error preparing HOG dataset"
    exit 1
fi

# Step 2: Prepare BRISK dataset
echo -e "\n${BLUE}Step 2: Preparing BRISK dataset...${NC}"
python scripts/prepare_dataset.py \
    --images $IMAGES_DIR \
    --annotations $ANNOTATIONS_DIR \
    --feature-type brisk \
    --output data/processed/detection_brisk.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BRISK dataset prepared${NC}"
else
    echo "Error preparing BRISK dataset"
    exit 1
fi

# Step 3: Train HOG model
echo -e "\n${BLUE}Step 3: Training HOG model...${NC}"
python scripts/train.py \
    --config configs/hog_config.yaml \
    --dataset data/processed/detection_hog.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ HOG model trained${NC}"
else
    echo "Error training HOG model"
    exit 1
fi

# Step 4: Train BRISK model
echo -e "\n${BLUE}Step 4: Training BRISK model...${NC}"
python scripts/train.py \
    --config configs/brisk_config.yaml \
    --dataset data/processed/detection_brisk.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BRISK model trained${NC}"
else
    echo "Error training BRISK model"
    exit 1
fi

# Step 5: Evaluate HOG model
echo -e "\n${BLUE}Step 5: Evaluating HOG model...${NC}"
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --dataset data/processed/detection_hog.pkl \
    --feature-type hog

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ HOG model evaluated${NC}"
else
    echo "Error evaluating HOG model"
    exit 1
fi

# Step 6: Evaluate BRISK model
echo -e "\n${BLUE}Step 6: Evaluating BRISK model...${NC}"
python scripts/evaluate.py \
    --model models/detection_brisk.h5 \
    --dataset data/processed/detection_brisk.pkl \
    --feature-type brisk

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ BRISK model evaluated${NC}"
else
    echo "Error evaluating BRISK model"
    exit 1
fi

echo -e "\n${GREEN}=================================="
echo "Pipeline completed successfully!"
echo "==================================${NC}"
echo ""
echo "Models saved in: models/"
echo "Results saved in: results/"
echo "Logs saved in: logs/"
echo ""
echo "Run inference with:"
echo "  python scripts/inference.py --model models/detection_hog.h5 --image test.jpg --feature-type hog"
echo ""
