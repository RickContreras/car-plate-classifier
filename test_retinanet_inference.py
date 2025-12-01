"""
Script rápido para probar la inferencia de RetinaNet
"""
import cv2
import numpy as np
from src.models.retinanet.inference import RetinaNetInference
import time

def main():
    print("="*70)
    print("PRUEBA DE INFERENCIA RETINANET")
    print("="*70)
    
    # Cargar modelo
    model_path = 'models/checkpoints/retinanet/retinanet_plates_best.h5'
    
    print(f"\n🔄 Inicializando modelo...")
    inference = RetinaNetInference(
        model_path=model_path,
        input_shape=(640, 640, 3),
        num_classes=1,
        confidence_threshold=0.01,  # Umbral muy bajo para debugging
        nms_threshold=0.5
    )
    
    # Probar con imágenes de ejemplo
    test_images = [
        'data/raw/images/Cars0.png',
        'data/raw/images/Cars1.png',
        'data/raw/images/Cars2.png',
    ]
    
    print("\n" + "="*70)
    print("PROBANDO DETECCIONES")
    print("="*70)
    
    for img_path in test_images:
        print(f"\n📸 Procesando: {img_path}")
        
        # Cargar imagen
        image = cv2.imread(img_path)
        if image is None:
            print(f"   ❌ No se pudo cargar la imagen")
            continue
        
        # Medir tiempo
        start = time.time()
        
        # Detectar
        box, score = inference.predict_single_best(image)
        
        elapsed = (time.time() - start) * 1000
        
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            print(f"   ✅ Detección encontrada:")
            print(f"      - Bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"      - Score: {score:.4f} ({score*100:.1f}%)")
            print(f"      - Tiempo: {elapsed:.1f} ms")
            
            # Dibujar y guardar
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                image, 
                f"RetinaNet: {score*100:.1f}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            
            # Guardar resultado
            output_path = img_path.replace('raw/images/', 'processed/retinanet_test_')
            cv2.imwrite(output_path, image)
            print(f"      - Guardado en: {output_path}")
        else:
            print(f"   ⚠️  No se detectó ninguna placa")
            print(f"      - Tiempo: {elapsed:.1f} ms")
    
    print("\n" + "="*70)
    print("✅ Prueba completada!")
    print("="*70)

if __name__ == "__main__":
    main()
