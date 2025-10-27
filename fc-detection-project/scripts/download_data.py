"""
Script para descargar el dataset de detección de placas desde Kaggle.
Dataset: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
"""

import os
import shutil
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Obtiene la ruta raíz del proyecto."""
    return Path(__file__).parent.parent


def setup_kaggle_credentials() -> bool:
    """
    Configura las credenciales de Kaggle desde el archivo en la raíz del proyecto.
    
    Returns:
        bool: True si las credenciales se configuraron correctamente, False en caso contrario.
    """
    project_root = get_project_root()
    kaggle_json_project = project_root / "kaggle.json"
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_home = kaggle_dir / "kaggle.json"
    
    # Verificar si existe kaggle.json en la raíz del proyecto
    if not kaggle_json_project.exists():
        print("❌ No se encontró kaggle.json en la raíz del proyecto")
        print(f"📁 Esperado en: {kaggle_json_project}")
        print("\n💡 Pasos para configurar:")
        print("   1. Ve a: https://www.kaggle.com/settings/account")
        print("   2. En la sección 'API', haz clic en 'Create New Token'")
        print("   3. Guarda el archivo kaggle.json en la raíz del proyecto")
        return False
    
    print(f"✅ kaggle.json encontrado en: {kaggle_json_project}")
    
    # Crear directorio .kaggle en home si no existe
    kaggle_dir.mkdir(exist_ok=True, parents=True)
    
    # Copiar kaggle.json al directorio home
    try:
        shutil.copy2(kaggle_json_project, kaggle_json_home)
        print(f"✅ Credenciales copiadas a: {kaggle_dir}")
    except Exception as e:
        print(f"⚠️  Advertencia al copiar credenciales: {e}")
    
    # Establecer permisos en Unix/Linux/Mac
    if os.name != 'nt':
        try:
            os.chmod(kaggle_json_home, 0o600)
            print("✅ Permisos configurados correctamente")
        except Exception as e:
            print(f"⚠️  Advertencia al establecer permisos: {e}")
    
    return True


def download_kaggle_dataset() -> None:
    """
    Descarga el dataset de detección de placas de vehículos desde Kaggle.
    
    Raises:
        ImportError: Si kaggle no está instalado.
        Exception: Si ocurre un error durante la descarga.
    """
    # Verificar y configurar credenciales
    if not setup_kaggle_credentials():
        print("\n❌ No se pudo configurar kaggle. Abortando descarga.")
        return
    
    # Importar kaggle después de configurar credenciales
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("\n❌ Error: kaggle no está instalado")
        print("💡 Instala con: pip install kaggle")
        print("   O ejecuta: pip install -r requirements.txt")
        return
    
    # Definir rutas
    project_root = get_project_root()
    data_raw_path = project_root / "data" / "raw"
    
    # Crear directorio si no existe
    data_raw_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("🔍 DESCARGANDO DATASET DE KAGGLE")
    print("=" * 60)
    print("📊 Dataset: andrewmvd/car-plate-detection")
    print(f"📁 Destino: {data_raw_path}")
    print()
    
    try:
        # Autenticar y descargar dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset='andrewmvd/car-plate-detection',
            path=str(data_raw_path),
            unzip=True,
            quiet=False
        )
        
        print("\n" + "=" * 60)
        print("✅ DATASET DESCARGADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📁 Ubicación: {data_raw_path.absolute()}")
        
        # Listar archivos descargados con tamaños
        print("\n📋 Archivos descargados:")
        total_size = 0
        file_count = 0
        
        for file in sorted(data_raw_path.rglob("*")):
            if file.is_file():
                size_bytes = file.stat().st_size
                size_mb = size_bytes / (1024 * 1024)
                total_size += size_bytes
                file_count += 1
                
                # Mostrar ruta relativa al directorio raw
                relative_path = file.relative_to(data_raw_path)
                print(f"   📄 {relative_path} ({size_mb:.2f} MB)")
        
        print(f"\n📊 Total: {file_count} archivos ({total_size / (1024 * 1024):.2f} MB)")
        print("=" * 60)
            
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR AL DESCARGAR EL DATASET")
        print("=" * 60)
        print(f"Error: {e}")
        print("\n💡 Verifica:")
        print("   1. Que kaggle.json tenga las credenciales correctas")
        print("   2. Que tengas acceso al dataset en Kaggle")
        print("   3. Tu conexión a internet")
        print("   4. Que el dataset existe: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection")
        raise


def main() -> None:
    """Función principal."""
    try:
        download_kaggle_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Descarga cancelada por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        exit(1)


if __name__ == "__main__":
    main()