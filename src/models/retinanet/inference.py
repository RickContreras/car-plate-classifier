"""
Módulo de inferencia para RetinaNet
Facilita el uso del modelo para detección de placas
"""
import numpy as np
import cv2
import tensorflow as tf
from typing import List, Tuple, Optional
from .anchors import AnchorGenerator
from .detector import RetinaNetDetector


class RetinaNetInference:
    """
    Clase para realizar inferencia con RetinaNet de forma simplificada
    """
    
    def __init__(
        self,
        model_path: str,
        input_shape: Tuple[int, int, int] = (640, 640, 3),
        num_classes: int = 1,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ):
        """
        Inicializar módulo de inferencia
        
        Args:
            model_path: Ruta al modelo entrenado (.h5)
            input_shape: Forma de entrada esperada [H, W, C]
            num_classes: Número de clases (1 para placas)
            confidence_threshold: Umbral de confianza para detecciones
            nms_threshold: Umbral para Non-Maximum Suppression
            max_detections: Número máximo de detecciones por imagen
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Cargar modelo
        print(f"🔄 Cargando modelo desde {model_path}...")
        
        # Intentar cargar directamente (si fue guardado sin wrapper)
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            print("✅ Modelo cargado directamente!")
        except Exception as e:
            # Si falla, construir arquitectura y cargar pesos
            print(f"⚠️ Carga directa falló: {str(e)[:100]}")
            print("🔄 Construyendo arquitectura y cargando pesos...")
            
            from .detector import RetinaNetDetector
            
            detector = RetinaNetDetector(
                num_classes=num_classes,
                input_shape=input_shape,
                backbone_type='resnet50',
                backbone_weights=None
            )
            
            self.model = detector.build()
            self.model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print("✅ Modelo construido y pesos cargados!")
        
        print(f"📊 Modelo tiene {len(self.model.layers)} capas")
        
        # Crear generador de anchors para decodificar predicciones
        self.anchor_generator = AnchorGenerator(
            sizes=[32, 64, 128, 256, 512],
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[1.0, 1.26, 1.59]
        )
        
        # Generar anchors para el tamaño de entrada
        dummy_image = np.zeros((1, *input_shape))
        self.anchors = self.anchor_generator.generate_anchors(
            image_shape=input_shape[:2]
        )
        print(f"📍 {len(self.anchors)} anchors generados")
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesar imagen para inferencia
        
        Args:
            image: Imagen BGR (OpenCV format)
        
        Returns:
            imagen procesada [1, H, W, 3], tamaño original (H, W)
        """
        # Guardar tamaño original
        h_orig, w_orig = image.shape[:2]
        
        # Resize a tamaño de entrada
        h_target, w_target = self.input_shape[:2]
        image_resized = cv2.resize(image, (w_target, h_target))
        
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar a [0, 1]
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        image_batch = np.expand_dims(image_norm, axis=0)
        
        return image_batch, (h_orig, w_orig)
    
    def decode_predictions(
        self,
        cls_preds: np.ndarray,
        box_preds: np.ndarray,
        anchors: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decodificar predicciones del modelo a bounding boxes
        
        Args:
            cls_preds: Predicciones de clasificación [num_anchors, num_classes]
            box_preds: Predicciones de regresión [num_anchors, 4]
            anchors: Anchors [num_anchors, 4]
            original_size: Tamaño original de la imagen (H, W)
        
        Returns:
            boxes: [N, 4] coordenadas (x1, y1, x2, y2)
            scores: [N] scores de confianza
            classes: [N] índices de clase
        """
        # Aplicar sigmoid a las predicciones de clasificación
        scores = tf.nn.sigmoid(cls_preds).numpy()
        
        # Para cada clase, encontrar detecciones válidas
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for class_idx in range(self.num_classes):
            class_scores = scores[:, class_idx]
            
            # Filtrar por umbral de confianza
            valid_mask = class_scores >= self.confidence_threshold
            valid_scores = class_scores[valid_mask]
            valid_boxes = box_preds[valid_mask]
            valid_anchors = anchors[valid_mask]
            
            if len(valid_scores) == 0:
                continue
            
            # Decodificar boxes desde deltas
            decoded_boxes = self._decode_boxes(valid_boxes, valid_anchors)
            
            # Aplicar NMS
            selected_indices = tf.image.non_max_suppression(
                boxes=decoded_boxes,
                scores=valid_scores,
                max_output_size=self.max_detections,
                iou_threshold=self.nms_threshold
            ).numpy()
            
            # Recopilar detecciones después de NMS
            all_boxes.append(decoded_boxes[selected_indices])
            all_scores.append(valid_scores[selected_indices])
            all_classes.append(np.full(len(selected_indices), class_idx))
        
        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Concatenar todas las detecciones
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        # Escalar boxes al tamaño original
        h_orig, w_orig = original_size
        h_model, w_model = self.input_shape[:2]
        
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * w_orig / w_model
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * h_orig / h_model
        
        # Limitar a máximo de detecciones
        if len(boxes) > self.max_detections:
            top_indices = np.argsort(scores)[::-1][:self.max_detections]
            boxes = boxes[top_indices]
            scores = scores[top_indices]
            classes = classes[top_indices]
        
        return boxes, scores, classes
    
    def _decode_boxes(
        self,
        box_deltas: np.ndarray,
        anchors: np.ndarray
    ) -> np.ndarray:
        """
        Decodificar deltas de cajas a coordenadas absolutas
        
        Args:
            box_deltas: [N, 4] deltas predichos (dy, dx, dh, dw)
            anchors: [N, 4] anchors (y1, x1, y2, x2)
        
        Returns:
            boxes: [N, 4] cajas decodificadas (x1, y1, x2, y2)
        """
        # Convertir anchors de esquinas a centro + tamaño
        anchor_heights = anchors[:, 2] - anchors[:, 0]
        anchor_widths = anchors[:, 3] - anchors[:, 1]
        anchor_cy = anchors[:, 0] + 0.5 * anchor_heights
        anchor_cx = anchors[:, 1] + 0.5 * anchor_widths
        
        # Aplicar deltas
        dy, dx, dh, dw = box_deltas[:, 0], box_deltas[:, 1], box_deltas[:, 2], box_deltas[:, 3]
        
        cy = dy * anchor_heights + anchor_cy
        cx = dx * anchor_widths + anchor_cx
        h = np.exp(dh) * anchor_heights
        w = np.exp(dw) * anchor_widths
        
        # Convertir de centro + tamaño a esquinas (x1, y1, x2, y2)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        return boxes
    
    def predict(
        self,
        image: np.ndarray,
        return_scores: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detectar placas en una imagen
        
        Args:
            image: Imagen BGR (OpenCV format)
            return_scores: Si retornar scores y clases
        
        Returns:
            boxes: [N, 4] bounding boxes (x1, y1, x2, y2)
            scores: [N] scores de confianza (si return_scores=True)
            classes: [N] índices de clase (si return_scores=True)
        """
        # Preprocesar
        image_batch, original_size = self.preprocess_image(image)
        
        # Inferencia
        predictions = self.model.predict(image_batch, verbose=0)
        
        # Las predicciones son una lista de tensors para cada nivel de FPN
        # Necesitamos concatenarlos
        if isinstance(predictions, list):
            # Predictions es [cls_outputs, box_outputs]
            cls_preds = predictions[0][0]  # [num_anchors, num_classes]
            box_preds = predictions[1][0]  # [num_anchors, 4]
        else:
            # Si es un solo tensor (modelo compilado diferente)
            cls_preds = predictions[0, :, :self.num_classes]
            box_preds = predictions[0, :, self.num_classes:]
        
        # Decodificar predicciones
        boxes, scores, classes = self.decode_predictions(
            cls_preds=cls_preds,
            box_preds=box_preds,
            anchors=self.anchors,
            original_size=original_size
        )
        
        if return_scores:
            return boxes, scores, classes
        else:
            return boxes, None, None
    
    def predict_single_best(
        self,
        image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Detectar la mejor placa (mayor score) en una imagen
        
        Args:
            image: Imagen BGR (OpenCV format)
        
        Returns:
            box: [4] bounding box (x1, y1, x2, y2) o None si no hay detecciones
            score: score de confianza o None si no hay detecciones
        """
        boxes, scores, _ = self.predict(image, return_scores=True)
        
        if len(boxes) == 0:
            return None, None
        
        # Tomar la detección con mayor score
        best_idx = np.argmax(scores)
        return boxes[best_idx], scores[best_idx]
