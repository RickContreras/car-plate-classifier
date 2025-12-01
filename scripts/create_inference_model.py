"""
Script para crear un modelo RetinaNet limpio solo para inferencia
desde el checkpoint entrenado
"""
import tensorflow as tf
from tensorflow import keras
import h5py
import sys

def create_inference_model(checkpoint_path, output_path):
    """
    Extraer el modelo base del checkpoint y guardarlo para inferencia
    """
    print(f"🔄 Leyendo checkpoint: {checkpoint_path}")
    
    # Importar arquitectura
    from src.models.retinanet.detector import RetinaNetDetector
    
    # Construir modelo
    print("🔄 Construyendo arquitectura...")
    detector = RetinaNetDetector(
        num_classes=1,
        input_shape=(640, 640, 3),
        backbone_type='resnet50',
        backbone_weights=None
    )
    
    model = detector.build()
    print(f"✅ Modelo construido: {len(model.layers)} capas")
    
    # Cargar pesos desde el checkpoint
    print("🔄 Cargando pesos del checkpoint...")
    
    # El checkpoint tiene estructura: model_weights/retinanet_plates/{layer_name}
    # Necesitamos mapear a nuestro modelo
    with h5py.File(checkpoint_path, 'r') as f:
        weight_group = f['model_weights']['retinanet_plates']
        
        loaded_layers = 0
        for layer in model.layers:
            if layer.name in weight_group:
                layer_group = weight_group[layer.name]
                weights = []
                
                # Cargar pesos de la capa
                weight_names = list(layer_group.keys())
                for wn in sorted(weight_names):
                    weights.append(layer_group[wn][:])
                
                if weights:
                    try:
                        layer.set_weights(weights)
                        loaded_layers += 1
                    except Exception as e:
                        print(f"   ⚠️ Error en capa {layer.name}: {e}")
        
        print(f"✅ Pesos cargados: {loaded_layers}/{len(model.layers)} capas")
    
    # Guardar modelo limpio
    print(f"🔄 Guardando modelo de inferencia: {output_path}")
    model.save(output_path, save_format='h5', include_optimizer=False)
    print("✅ Modelo guardado!")
    
    return model

if __name__ == "__main__":
    checkpoint = "models/checkpoints/retinanet/retinanet_plates_best.h5"
    output = "models/retinanet_inference.h5"
    
    model = create_inference_model(checkpoint, output)
    
    print("\n" + "="*70)
    print("✅ Modelo de inferencia creado exitosamente!")
    print(f"   📁 Guardado en: {output}")
    print("="*70)
