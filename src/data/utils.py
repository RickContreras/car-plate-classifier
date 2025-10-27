"""
Utilidades de dataset y cargadores para detección.
"""

import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Optional
import pickle


def normalize_bbox(bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Normalizar coordenadas de bounding box a [0, 1].
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) en píxeles
        img_width: Ancho de imagen
        img_height: Alto de imagen
        
    Returns:
        (x_center, y_center, width, height) normalizado a [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Convertir a formato centro
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalizar
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)


def denormalize_bbox(norm_bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Desnormalizar coordenadas de bounding box de [0, 1] a píxeles.
    
    Args:
        norm_bbox: (x_center, y_center, width, height) normalizado
        img_width: Ancho de imagen
        img_height: Alto de imagen
        
    Returns:
        (xmin, ymin, xmax, ymax) en píxeles
    """
    x_center, y_center, width, height = norm_bbox
    
    # Desnormalizar
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convertir a formato de esquinas
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)
    
    # Recortar a límites de imagen
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    
    return (xmin, ymin, xmax, ymax)


def parse_pascal_voc(xml_path: str) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Parsear archivo de anotación XML Pascal VOC.
    
    Args:
        xml_path: Ruta al archivo XML
        
    Returns:
        Tupla de (filename, lista de bboxes)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    
    bboxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return filename, bboxes


class DetectionDataset:
    """Clase de dataset para detección con extracción de características."""
    
    def __init__(
        self,
        features: np.ndarray,
        bboxes: np.ndarray,
        image_paths: Optional[List[str]] = None
    ):
        """
        Inicializar dataset de detección.
        
        Args:
            features: Array de forma (n_samples, feature_dim)
            bboxes: Array de forma (n_samples, 4) - coordenadas normalizadas
            image_paths: Lista opcional de rutas de imágenes
        """
        assert len(features) == len(bboxes), "Features y bboxes deben tener la misma longitud"
        assert bboxes.shape[1] == 4, "Bboxes debe tener forma (n_samples, 4)"
        
        self.features = features
        self.bboxes = bboxes
        self.image_paths = image_paths
        self.n_samples = len(features)
    
    def __len__(self) -> int:
        """Obtener tamaño del dataset."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Obtener una muestra individual."""
        return self.features[idx], self.bboxes[idx]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Obtener un batch de muestras."""
        return self.features[indices], self.bboxes[indices]
    
    def split(self, train_ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None) -> Tuple['DetectionDataset', 'DetectionDataset']:
        """
        Dividir dataset en conjuntos de entrenamiento y validación.
        
        Args:
            train_ratio: Proporción de muestras de entrenamiento
            shuffle: Si se debe mezclar antes de dividir
            seed: Semilla aleatoria
            
        Returns:
            Tupla de (train_dataset, val_dataset)
        """
        if seed is not None:
            np.random.seed(seed)
        
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(train_ratio * self.n_samples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_paths = [self.image_paths[i] for i in train_indices] if self.image_paths else None
        val_paths = [self.image_paths[i] for i in val_indices] if self.image_paths else None
        
        train_dataset = DetectionDataset(
            features=self.features[train_indices],
            bboxes=self.bboxes[train_indices],
            image_paths=train_paths
        )
        
        val_dataset = DetectionDataset(
            features=self.features[val_indices],
            bboxes=self.bboxes[val_indices],
            image_paths=val_paths
        )
        
        return train_dataset, val_dataset
    
    def save(self, filepath: str):
        """Guardar dataset a archivo pickle."""
        data = {
            'features': self.features,
            'bboxes': self.bboxes,
            'image_paths': self.image_paths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath: str) -> 'DetectionDataset':
        """Cargar dataset desde archivo pickle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return DetectionDataset(
            features=data['features'],
            bboxes=data['bboxes'],
            image_paths=data.get('image_paths')
        )


def load_dataset(filepath: str) -> DetectionDataset:
    """
    Cargar dataset desde archivo.
    
    Args:
        filepath: Ruta al archivo de dataset (.pkl)
        
    Returns:
        Instancia de DetectionDataset
    """
    return DetectionDataset.load(filepath)
