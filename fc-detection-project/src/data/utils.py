"""
Dataset utilities and loaders for detection.
"""

import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, List, Optional
import pickle


def normalize_bbox(bbox: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates to [0, 1].
    
    Args:
        bbox: (xmin, ymin, xmax, ymax) in pixels
        img_width: Image width
        img_height: Image height
        
    Returns:
        (x_center, y_center, width, height) normalized to [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Convert to center format
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)


def denormalize_bbox(norm_bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Denormalize bounding box coordinates from [0, 1] to pixels.
    
    Args:
        norm_bbox: (x_center, y_center, width, height) normalized
        img_width: Image width
        img_height: Image height
        
    Returns:
        (xmin, ymin, xmax, ymax) in pixels
    """
    x_center, y_center, width, height = norm_bbox
    
    # Denormalize
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    # Convert to corner format
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)
    
    # Clip to image bounds
    xmin = max(0, min(xmin, img_width - 1))
    ymin = max(0, min(ymin, img_height - 1))
    xmax = max(0, min(xmax, img_width - 1))
    ymax = max(0, min(ymax, img_height - 1))
    
    return (xmin, ymin, xmax, ymax)


def parse_pascal_voc(xml_path: str) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    """
    Parse Pascal VOC XML annotation file.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        Tuple of (filename, list of bboxes)
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
    """Dataset class for detection with feature extraction."""
    
    def __init__(
        self,
        features: np.ndarray,
        bboxes: np.ndarray,
        image_paths: Optional[List[str]] = None
    ):
        """
        Initialize detection dataset.
        
        Args:
            features: Array of shape (n_samples, feature_dim)
            bboxes: Array of shape (n_samples, 4) - normalized coordinates
            image_paths: Optional list of image paths
        """
        assert len(features) == len(bboxes), "Features and bboxes must have same length"
        assert bboxes.shape[1] == 4, "Bboxes must have shape (n_samples, 4)"
        
        self.features = features
        self.bboxes = bboxes
        self.image_paths = image_paths
        self.n_samples = len(features)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single sample."""
        return self.features[idx], self.bboxes[idx]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples."""
        return self.features[indices], self.bboxes[indices]
    
    def split(self, train_ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None) -> Tuple['DetectionDataset', 'DetectionDataset']:
        """
        Split dataset into train and validation sets.
        
        Args:
            train_ratio: Ratio of training samples
            shuffle: Whether to shuffle before splitting
            seed: Random seed
            
        Returns:
            Tuple of (train_dataset, val_dataset)
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
        """Save dataset to pickle file."""
        data = {
            'features': self.features,
            'bboxes': self.bboxes,
            'image_paths': self.image_paths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath: str) -> 'DetectionDataset':
        """Load dataset from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return DetectionDataset(
            features=data['features'],
            bboxes=data['bboxes'],
            image_paths=data.get('image_paths')
        )


def load_dataset(filepath: str) -> DetectionDataset:
    """
    Load dataset from file.
    
    Args:
        filepath: Path to dataset file (.pkl)
        
    Returns:
        DetectionDataset instance
    """
    return DetectionDataset.load(filepath)
