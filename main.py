#!/usr/bin/env python3
"""
Script principal para entrenar el clasificador de placas vehiculares.
Ejecuta todo el pipeline: preprocesamiento, extracción de características y entrenamiento.
"""

import os
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import prepare_dataset, load_config
from src.feature_extraction import extract_all_features
from src.train_models import train_and_evaluate_all_models


def main():
    """Función principal del pipeline de entrenamiento."""
    
    print("\n" + "="*70)
    print(" "*15 + "🚗 CLASIFICADOR DE PLACAS VEHICULARES 🚗")
    print("="*70)
    
    try:
        # 1. Cargar configuración
        print("\n📁 Cargando configuración...")
        config = load_config()
        print("✅ Configuración cargada")
        
        # 2. Preparar dataset
        print("\n" + "="*70)
        print("PASO 1: PREPROCESAMIENTO DE DATOS")
        print("="*70)
        X_train, X_test, y_train, y_test = prepare_dataset(config)
        
        # 3. Extraer características
        print("\n" + "="*70)
        print("PASO 2: EXTRACCIÓN DE CARACTERÍSTICAS")
        print("="*70)
        train_features, test_features = extract_all_features(X_train, X_test, config)
        
        # 4. Entrenar modelos
        print("\n" + "="*70)
        print("PASO 3: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("="*70)
        results, models = train_and_evaluate_all_models(
            train_features, test_features,
            y_train, y_test,
            config
        )
        
        # 5. Resumen final
        print("\n" + "="*70)
        print("🎉 PROCESO COMPLETADO EXITOSAMENTE 🎉")
        print("="*70)
        
        print("\n📊 Modelos entrenados:")
        for model_name in results.keys():
            print(f"   • {model_name}")
        
        print(f"\n📂 Modelos guardados en: {config['model']['save_path']}")
        print(f"📂 Resultados guardados en: {config['evaluation']['results_path']}")
        
        print("\n💡 Siguiente paso:")
        print("   Ejecuta 'python app/gui.py' para usar la interfaz gráfica")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: No se encontró el archivo o directorio: {e}")
        print("\n💡 Asegúrate de:")
        print("   1. Haber descargado el dataset usando 'python scripts/download_data.py'")
        print("   2. Que existan las carpetas 'data/raw/images' y 'data/raw/annotations'")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n" + "="*70)


if __name__ == "__main__":
    main()
