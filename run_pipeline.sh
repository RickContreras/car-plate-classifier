#!/bin/bash
# Ejemplo de pipeline completo para FC Detection Project

echo "=================================="
echo "FC Detection - Pipeline Completo"
echo "=================================="

# Configuración
IMAGES_DIR="data/raw/images"
ANNOTATIONS_DIR="data/raw/annotations"

# Colores para salida
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # Sin Color

# Paso 1: Preparar dataset HOG
echo -e "\n${BLUE}Paso 1: Preparando dataset HOG...${NC}"
python scripts/prepare_dataset.py \
    --images $IMAGES_DIR \
    --annotations $ANNOTATIONS_DIR \
    --feature-type hog \
    --output data/processed/detection_hog.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset HOG preparado${NC}"
else
    echo "Error preparando dataset HOG"
    exit 1
fi

# Paso 2: Preparar dataset BRISK
echo -e "\n${BLUE}Paso 2: Preparando dataset BRISK...${NC}"
python scripts/prepare_dataset.py \
    --images $IMAGES_DIR \
    --annotations $ANNOTATIONS_DIR \
    --feature-type brisk \
    --output data/processed/detection_brisk.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset BRISK preparado${NC}"
else
    echo "Error preparando dataset BRISK"
    exit 1
fi

# Paso 3: Entrenar modelo HOG
echo -e "\n${BLUE}Paso 3: Entrenando modelo HOG...${NC}"
python scripts/train.py \
    --config configs/hog_config.yaml \
    --dataset data/processed/detection_hog.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Modelo HOG entrenado${NC}"
else
    echo "Error entrenando modelo HOG"
    exit 1
fi

# Paso 4: Entrenar modelo BRISK
echo -e "\n${BLUE}Paso 4: Entrenando modelo BRISK...${NC}"
python scripts/train.py \
    --config configs/brisk_config.yaml \
    --dataset data/processed/detection_brisk.pkl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Modelo BRISK entrenado${NC}"
else
    echo "Error entrenando modelo BRISK"
    exit 1
fi

# Paso 5: Evaluar modelo HOG
echo -e "\n${BLUE}Paso 5: Evaluando modelo HOG...${NC}"
python scripts/evaluate.py \
    --model models/detection_hog.h5 \
    --dataset data/processed/detection_hog.pkl \
    --feature-type hog

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Modelo HOG evaluado${NC}"
else
    echo "Error evaluando modelo HOG"
    exit 1
fi

# Paso 6: Evaluar modelo BRISK
echo -e "\n${BLUE}Paso 6: Evaluando modelo BRISK...${NC}"
python scripts/evaluate.py \
    --model models/detection_brisk.h5 \
    --dataset data/processed/detection_brisk.pkl \
    --feature-type brisk

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Modelo BRISK evaluado${NC}"
else
    echo "Error evaluando modelo BRISK"
    exit 1
fi

echo -e "\n${GREEN}=================================="
echo "¡Pipeline completado exitosamente!"
echo "==================================${NC}"
echo ""
echo "Modelos guardados en: models/"
echo "Resultados guardados en: results/"
echo "Logs guardados en: logs/"
echo ""
echo "Ejecutar inferencia con:"
echo "  python scripts/inference.py --model models/detection_hog.h5 --image test.jpg --feature-type hog"
echo ""
