"""
Dataset personalizado para entrenamiento de RetinaNet.

Este módulo implementa el pipeline de datos para RetinaNet, incluyendo:
- Carga y preprocesamiento de imágenes
- Generación de targets (clasificación + regresión) para anchors
- Data augmentation
- Batching eficiente
"""

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import xml.etree.ElementTree as ET

from ..models.retinanet.anchors import AnchorGenerator, encode_boxes


class RetinaNetDataset:
    """
    Dataset para entrenamiento de RetinaNet.
    
    A diferencia de DetectionDataset (que usa features pre-extraídas),
    este dataset trabaja directamente con imágenes y genera targets
    para el entrenamiento end-to-end de RetinaNet.
    
    Attributes:
        image_paths: Lista de rutas a imágenes
        annotations: Lista de bboxes por imagen
        image_shape: Forma objetivo de imágenes (H, W)
        anchor_generator: Generador de anchor boxes
        iou_threshold_pos: Umbral IoU para anchors positivas
        iou_threshold_neg: Umbral IoU para anchors negativas
    """
    
    def __init__(
        self,
        image_paths: List[str],
        annotations: List[List[Tuple[int, int, int, int]]],
        image_shape: Tuple[int, int] = (640, 640),
        anchor_generator: Optional[AnchorGenerator] = None,
        iou_threshold_pos: float = 0.5,
        iou_threshold_neg: float = 0.4,
        augment: bool = False
    ):
        """
        Inicializar dataset de RetinaNet.
        
        Args:
            image_paths: Lista de rutas a imágenes
            annotations: Lista de listas de bboxes (xmin, ymin, xmax, ymax)
            image_shape: Forma objetivo (H, W) para resize
            anchor_generator: Generador de anchors (usa default si None)
            iou_threshold_pos: IoU >= este valor → anchor positiva
            iou_threshold_neg: IoU < este valor → anchor negativa
            augment: Si True, aplicar data augmentation
        """
        assert len(image_paths) == len(annotations), \
            "Número de imágenes y anotaciones debe coincidir"
        
        self.image_paths = image_paths
        self.annotations = annotations
        self.image_shape = image_shape
        self.iou_threshold_pos = iou_threshold_pos
        self.iou_threshold_neg = iou_threshold_neg
        self.augment = augment
        
        # Inicializar generador de anchors
        if anchor_generator is None:
            self.anchor_generator = AnchorGenerator()
        else:
            self.anchor_generator = anchor_generator
        
        # Generar anchors una vez (son fijas para todas las imágenes)
        self.anchors = self.anchor_generator.generate_anchors(image_shape)
        self.num_anchors = self.anchors.shape[0]
    
    def __len__(self) -> int:
        """Retornar número de muestras."""
        return len(self.image_paths)
    
    def _load_and_preprocess_image(
        self,
        image_path: str
    ) -> np.ndarray:
        """
        Cargar y preprocesar imagen.
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            Imagen preprocesada (H, W, 3) normalizada
        """
        # Leer imagen
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")
        
        # Convertir BGR a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        
        # Normalizar a [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Aplicar normalización ImageNet (usado en backbones pre-entrenados)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        return img
    
    def _compute_iou(
        self,
        boxes1: tf.Tensor,
        boxes2: tf.Tensor
    ) -> tf.Tensor:
        """
        Calcular IoU entre dos conjuntos de boxes.
        
        Args:
            boxes1: Tensor (N, 4) en formato (x1, y1, x2, y2)
            boxes2: Tensor (M, 4) en formato (x1, y1, x2, y2)
            
        Returns:
            Matriz de IoU (N, M)
        """
        # Expandir dimensiones para broadcasting
        boxes1 = tf.expand_dims(boxes1, axis=1)  # (N, 1, 4)
        boxes2 = tf.expand_dims(boxes2, axis=0)  # (1, M, 4)
        
        # Calcular intersección
        x1 = tf.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = tf.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = tf.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = tf.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = tf.maximum(0.0, x2 - x1) * tf.maximum(0.0, y2 - y1)
        
        # Calcular áreas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Calcular unión
        union = area1 + area2 - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def _assign_anchors_to_gt(
        self,
        gt_boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Asignar anchors a ground truth boxes.
        
        Para cada anchor, determinar si es:
        - Positiva: IoU >= threshold_pos con algún GT box
        - Negativa: IoU < threshold_neg con todos los GT boxes
        - Ignorada: Entre threshold_neg y threshold_pos
        
        Args:
            gt_boxes: Lista de ground truth boxes (xmin, ymin, xmax, ymax)
            
        Returns:
            Tupla de (class_targets, box_targets, positive_mask)
        """
        # Inicializar arrays
        class_targets = np.zeros((self.num_anchors, 1), dtype=np.float32)
        box_targets = np.zeros((self.num_anchors, 4), dtype=np.float32)
        positive_mask = np.zeros(self.num_anchors, dtype=bool)
        
        if len(gt_boxes) == 0:
            # Sin objetos: todas las anchors son negativas
            return class_targets, box_targets, positive_mask
        
        # Convertir GT boxes a tensor
        gt_boxes_tensor = tf.constant(gt_boxes, dtype=tf.float32)
        
        # Calcular IoU entre anchors y GT boxes
        iou_matrix = self._compute_iou(self.anchors, gt_boxes_tensor)  # (num_anchors, num_gt)
        
        # Para cada anchor, encontrar el GT box con mayor IoU
        max_iou = tf.reduce_max(iou_matrix, axis=1)  # (num_anchors,)
        max_gt_idx = tf.argmax(iou_matrix, axis=1)  # (num_anchors,)
        
        # Convertir a numpy para manipulación
        max_iou = max_iou.numpy()
        max_gt_idx = max_gt_idx.numpy()
        
        # Asignar anchors positivas (IoU >= threshold_pos)
        positive_mask = max_iou >= self.iou_threshold_pos
        
        # Para anchors positivas, asignar clase y calcular targets de regresión
        if np.any(positive_mask):
            class_targets[positive_mask] = 1.0  # Clase 1 (placa)
            
            # Codificar offsets de bounding box
            positive_anchors = self.anchors[positive_mask]
            assigned_gt_boxes = tf.gather(gt_boxes_tensor, max_gt_idx[positive_mask])
            
            encoded_boxes = encode_boxes(assigned_gt_boxes, positive_anchors)
            box_targets[positive_mask] = encoded_boxes.numpy()
        
        # Anchors negativas tienen class_targets = 0 (ya inicializado)
        
        return class_targets, box_targets, positive_mask
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple]:
        """
        Obtener una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Tupla de (image, (class_targets, box_targets, positive_mask))
        """
        # Cargar imagen
        image = self._load_and_preprocess_image(self.image_paths[idx])
        
        # Obtener anotaciones (escalar a tamaño de imagen procesada)
        original_annotations = self.annotations[idx]
        
        # TODO: Aquí aplicar data augmentation si self.augment = True
        # (horizontal flip, brightness, contrast, etc.)
        
        # Asignar anchors a ground truth
        class_targets, box_targets, positive_mask = \
            self._assign_anchors_to_gt(original_annotations)
        
        return image, (class_targets, box_targets, positive_mask)
    
    def get_tf_dataset(
        self,
        batch_size: int = 8,
        shuffle: bool = True,
        buffer_size: int = 1000
    ) -> tf.data.Dataset:
        """
        Crear un tf.data.Dataset para entrenamiento eficiente.
        
        Args:
            batch_size: Tamaño del batch
            shuffle: Si se debe mezclar el dataset
            buffer_size: Tamaño del buffer para shuffle
            
        Returns:
            tf.data.Dataset preparado para entrenamiento
        """
        def generator():
            """Generador para tf.data.Dataset."""
            indices = np.arange(len(self))
            if shuffle:
                np.random.shuffle(indices)
            
            for idx in indices:
                image, (class_targets, box_targets, positive_mask) = self[idx]
                # Combinar targets en un único tensor para compatibilidad
                # Formato: [class_targets, box_targets, positive_mask_expanded]
                positive_mask_float = positive_mask.astype(np.float32)
                yield image, class_targets, box_targets, positive_mask_float
        
        # Crear dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.image_shape, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_anchors, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_anchors, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(self.num_anchors,), dtype=tf.float32),
            )
        )
        
        # Aplicar transformaciones
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        # Repetir infinitamente para evitar "End of sequence"
        dataset = dataset.repeat()
        
        # Batch y restructurar para que y sea una tupla
        def restructure_batch(images, cls_targets, box_targets, pos_masks):
            return images, (cls_targets, box_targets, pos_masks)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(restructure_batch)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @staticmethod
    def from_pascal_voc(
        images_dir: str,
        annotations_dir: str,
        image_shape: Tuple[int, int] = (640, 640),
        **kwargs
    ) -> 'RetinaNetDataset':
        """
        Crear dataset desde archivos Pascal VOC XML.
        
        Args:
            images_dir: Directorio con imágenes
            annotations_dir: Directorio con archivos XML
            image_shape: Forma objetivo de imágenes
            **kwargs: Argumentos adicionales para RetinaNetDataset
            
        Returns:
            Instancia de RetinaNetDataset
        """
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        
        image_paths = []
        annotations = []
        
        # Iterar sobre archivos XML
        for xml_file in sorted(annotations_path.glob('*.xml')):
            # Parsear XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            filename = root.find('filename').text
            image_file = images_path / filename
            
            if not image_file.exists():
                print(f"Advertencia: imagen no encontrada: {image_file}")
                continue
            
            # Extraer bounding boxes
            boxes = []
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                boxes.append((xmin, ymin, xmax, ymax))
            
            if boxes:  # Solo agregar si hay al menos una box
                image_paths.append(str(image_file))
                annotations.append(boxes)
        
        print(f"Cargadas {len(image_paths)} imágenes con anotaciones")
        
        return RetinaNetDataset(
            image_paths=image_paths,
            annotations=annotations,
            image_shape=image_shape,
            **kwargs
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple['RetinaNetDataset', 'RetinaNetDataset']:
        """
        Dividir dataset en train/val.
        
        Args:
            train_ratio: Proporción de entrenamiento
            shuffle: Si mezclar antes de dividir
            seed: Semilla aleatoria
            
        Returns:
            Tupla de (train_dataset, val_dataset)
        """
        if seed is not None:
            np.random.seed(seed)
        
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(train_ratio * len(self))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_dataset = RetinaNetDataset(
            image_paths=[self.image_paths[i] for i in train_indices],
            annotations=[self.annotations[i] for i in train_indices],
            image_shape=self.image_shape,
            anchor_generator=self.anchor_generator,
            iou_threshold_pos=self.iou_threshold_pos,
            iou_threshold_neg=self.iou_threshold_neg,
            augment=self.augment
        )
        
        val_dataset = RetinaNetDataset(
            image_paths=[self.image_paths[i] for i in val_indices],
            annotations=[self.annotations[i] for i in val_indices],
            image_shape=self.image_shape,
            anchor_generator=self.anchor_generator,
            iou_threshold_pos=self.iou_threshold_pos,
            iou_threshold_neg=self.iou_threshold_neg,
            augment=False  # Sin augmentation en validación
        )
        
        return train_dataset, val_dataset
