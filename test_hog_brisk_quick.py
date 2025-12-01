"""
Prueba rápida de HOG y BRISK funcionando correctamente
"""
import cv2
import numpy as np
from tensorflow import keras
from src.features.hog import HOGFeatureExtractor
from src.features.brisk import BRISKFeatureExtractor
from src.data.utils import denormalize_bbox

print("=" * 70)
print("PRUEBA RÁPIDA: HOG Y BRISK")
print("=" * 70)

# Cargar modelos
print("\n🔄 Cargando modelos...")
hog_model = keras.models.load_model('models/detection_hog_best.h5', compile=False)
brisk_model = keras.models.load_model('models/detection_brisk_best.h5', compile=False)
print("✅ Modelos cargados!")

# Cargar imagen
test_image = 'data/raw/images/Cars0.png'
print(f"\n📸 Cargando imagen: {test_image}")
image = cv2.imread(test_image)
if image is None:
    print("❌ Error: No se pudo cargar la imagen")
    exit(1)

h, w = image.shape[:2]
print(f"   Tamaño: {w}x{h}")

# Probar HOG
print("\n🟢 Probando HOG...")
hog_extractor = HOGFeatureExtractor()
hog_features = hog_extractor.extract(image).reshape(1, -1)
hog_pred = hog_model.predict(hog_features, verbose=0)[0]
hog_bbox = denormalize_bbox(hog_pred, w, h)
print(f"   ✅ Bbox HOG: {hog_bbox}")
print(f"   📊 Normalized: {hog_pred}")

# Probar BRISK
print("\n🔵 Probando BRISK...")
brisk_extractor = BRISKFeatureExtractor()
brisk_features = brisk_extractor.extract(image).reshape(1, -1)
brisk_pred = brisk_model.predict(brisk_features, verbose=0)[0]
brisk_bbox = denormalize_bbox(brisk_pred, w, h)
print(f"   ✅ Bbox BRISK: {brisk_bbox}")
print(f"   📊 Normalized: {brisk_pred}")

# Dibujar resultados
result_img = image.copy()
x1, y1, x2, y2 = hog_bbox
cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.putText(result_img, "HOG", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

x1, y1, x2, y2 = brisk_bbox
cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 165, 0), 2)
cv2.putText(result_img, "BRISK", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

# Guardar
output_path = 'data/processed/test_hog_brisk.png'
cv2.imwrite(output_path, result_img)
print(f"\n💾 Resultado guardado en: {output_path}")

print("\n" + "=" * 70)
print("✅ PRUEBA COMPLETADA EXITOSAMENTE!")
print("=" * 70)
