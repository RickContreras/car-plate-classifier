"""
Interfaz gráfica para el clasificador de placas vehiculares.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import yaml
import os
from pathlib import Path
import sys

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import preprocess_image
from src.feature_extraction import HOGFeatureExtractor, BRISKFeatureExtractor
from src.train_models import PlateClassifier


class PlateClassifierGUI:
    """Interfaz gráfica para clasificación de placas."""
    
    def __init__(self, root):
        """
        Inicializa la interfaz gráfica.
        
        Args:
            root: Ventana raíz de tkinter
        """
        self.root = root
        self.root.title("Clasificador de Placas Vehiculares")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.model = None
        self.feature_extractor = None
        self.config = None
        
        # Cargar configuración
        self.load_configuration()
        
        # Configurar estilos
        self.setup_styles()
        
        # Crear interfaz
        self.create_widgets()
        
    def load_configuration(self):
        """Carga la configuración del proyecto."""
        try:
            config_path = "config/config.yaml"
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar configuración: {e}")
            self.config = self.get_default_config()
    
    def get_default_config(self):
        """Retorna configuración por defecto."""
        return {
            'data': {'img_size': [128, 128]},
            'preprocessing': {
                'resize': True,
                'grayscale': True,
                'normalize': True,
                'equalize_hist': False
            },
            'features': {
                'hog': {
                    'orientations': 9,
                    'pixels_per_cell': [8, 8],
                    'cells_per_block': [2, 2]
                },
                'brisk': {
                    'threshold': 30,
                    'octaves': 3,
                    'pattern_scale': 1.0
                }
            },
            'model': {'save_path': 'models/'}
        }
    
    def setup_styles(self):
        """Configura los estilos de la interfaz."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='green', font=('Arial', 11, 'bold'))
        style.configure('Error.TLabel', foreground='red', font=('Arial', 11, 'bold'))
        style.configure('TButton', font=('Arial', 10))
    
    def create_widgets(self):
        """Crea los widgets de la interfaz."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title = ttk.Label(main_frame, text="🚗 Clasificador de Placas Vehiculares", 
                         style='Title.TLabel')
        title.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Panel izquierdo - Controles
        self.create_control_panel(main_frame)
        
        # Panel derecho - Visualización
        self.create_display_panel(main_frame)
        
        # Panel inferior - Resultados
        self.create_results_panel(main_frame)
    
    def create_control_panel(self, parent):
        """Crea el panel de controles."""
        control_frame = ttk.LabelFrame(parent, text="Controles", padding="10")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.N, tk.W, tk.E))
        
        # Selector de modelo
        ttk.Label(control_frame, text="Modelo:", style='Header.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        
        self.model_var = tk.StringVar()
        model_options = self.get_available_models()
        
        if model_options:
            self.model_var.set(model_options[0])
        
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var,
                                   values=model_options, state='readonly', width=30)
        model_combo.grid(row=1, column=0, pady=5)
        
        # Botón cargar modelo
        ttk.Button(control_frame, text="📂 Cargar Modelo", 
                  command=self.load_model).grid(row=2, column=0, pady=5)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=3, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        # Selector de tipo de característica
        ttk.Label(control_frame, text="Características:", style='Header.TLabel').grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        
        self.feature_var = tk.StringVar(value='hog')
        ttk.Radiobutton(control_frame, text="HOG", variable=self.feature_var,
                       value='hog').grid(row=5, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="BRISK", variable=self.feature_var,
                       value='brisk').grid(row=6, column=0, sticky=tk.W)
        
        # Separador
        ttk.Separator(control_frame, orient='horizontal').grid(
            row=7, column=0, sticky=(tk.W, tk.E), pady=10
        )
        
        # Botones de acción
        ttk.Button(control_frame, text="📷 Cargar Imagen", 
                  command=self.load_image).grid(row=8, column=0, pady=5)
        
        ttk.Button(control_frame, text="🔍 Clasificar", 
                  command=self.classify_image).grid(row=9, column=0, pady=5)
        
        ttk.Button(control_frame, text="🗑️ Limpiar", 
                  command=self.clear_all).grid(row=10, column=0, pady=5)
        
        # Estado del modelo
        ttk.Label(control_frame, text="Estado:", style='Header.TLabel').grid(
            row=11, column=0, sticky=tk.W, pady=(10, 5)
        )
        
        self.status_label = ttk.Label(control_frame, text="❌ Modelo no cargado",
                                     style='Error.TLabel')
        self.status_label.grid(row=12, column=0, sticky=tk.W)
    
    def create_display_panel(self, parent):
        """Crea el panel de visualización de imágenes."""
        display_frame = ttk.LabelFrame(parent, text="Visualización", padding="10")
        display_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.N, tk.S, tk.W, tk.E))
        
        parent.rowconfigure(1, weight=1)
        
        # Frame para imagen original
        original_frame = ttk.Frame(display_frame)
        original_frame.grid(row=0, column=0, padx=5)
        
        ttk.Label(original_frame, text="Imagen Original", 
                 style='Header.TLabel').pack()
        
        self.original_canvas = tk.Canvas(original_frame, width=350, height=350,
                                        bg='gray85', relief=tk.SUNKEN, borderwidth=2)
        self.original_canvas.pack(pady=5)
        
        # Frame para imagen procesada
        processed_frame = ttk.Frame(display_frame)
        processed_frame.grid(row=0, column=1, padx=5)
        
        ttk.Label(processed_frame, text="Imagen Procesada", 
                 style='Header.TLabel').pack()
        
        self.processed_canvas = tk.Canvas(processed_frame, width=350, height=350,
                                         bg='gray85', relief=tk.SUNKEN, borderwidth=2)
        self.processed_canvas.pack(pady=5)
    
    def create_results_panel(self, parent):
        """Crea el panel de resultados."""
        results_frame = ttk.LabelFrame(parent, text="Resultados", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5,
                          sticky=(tk.W, tk.E))
        
        self.result_label = ttk.Label(results_frame, text="",
                                     font=('Arial', 14, 'bold'))
        self.result_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(results_frame, text="",
                                         font=('Arial', 11))
        self.confidence_label.pack()
    
    def get_available_models(self):
        """Obtiene lista de modelos disponibles."""
        models_path = Path(self.config['model']['save_path'])
        
        if not models_path.exists():
            return []
        
        models = []
        for ext in ['.pkl', '.h5']:
            models.extend([f.stem for f in models_path.glob(f'*{ext}')])
        
        return sorted(set(models))
    
    def load_model(self):
        """Carga el modelo seleccionado."""
        model_name = self.model_var.get()
        
        if not model_name:
            messagebox.showwarning("Advertencia", "Seleccione un modelo primero")
            return
        
        try:
            # Determinar tipo de modelo y características
            parts = model_name.split('_')
            if len(parts) >= 2:
                model_type = '_'.join(parts[:-1])
                feature_type = parts[-1]
            else:
                messagebox.showerror("Error", "Nombre de modelo inválido")
                return
            
            # Cargar modelo
            models_path = Path(self.config['model']['save_path'])
            
            # Intentar con ambas extensiones
            model_path = models_path / f"{model_name}.pkl"
            if not model_path.exists():
                model_path = models_path / f"{model_name}.h5"
            
            if not model_path.exists():
                messagebox.showerror("Error", f"Modelo no encontrado: {model_name}")
                return
            
            self.model = PlateClassifier(model_type=model_type)
            self.model.load(str(model_path))
            
            # Configurar extractor de características
            if feature_type == 'hog':
                self.feature_extractor = HOGFeatureExtractor(self.config)
            elif feature_type == 'brisk':
                self.feature_extractor = BRISKFeatureExtractor(self.config)
            else:
                messagebox.showerror("Error", f"Tipo de característica desconocido: {feature_type}")
                return
            
            self.feature_var.set(feature_type)
            
            self.status_label.config(text=f"✅ Modelo cargado: {model_name}",
                                    style='Success.TLabel')
            messagebox.showinfo("Éxito", f"Modelo cargado correctamente:\n{model_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar modelo:\n{e}")
            self.status_label.config(text="❌ Error al cargar modelo",
                                    style='Error.TLabel')
    
    def load_image(self):
        """Carga una imagen desde el disco."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar Imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Cargar imagen
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            # Preprocesar imagen
            self.processed_image = preprocess_image(self.original_image, self.config)
            
            # Mostrar imágenes
            self.display_image(self.original_image, self.original_canvas)
            self.display_processed_image(self.processed_image, self.processed_canvas)
            
            # Limpiar resultados anteriores
            self.result_label.config(text="")
            self.confidence_label.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar imagen:\n{e}")
    
    def display_image(self, image, canvas):
        """Muestra una imagen en un canvas."""
        # Convertir de BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para ajustar al canvas
        h, w = image_rgb.shape[:2]
        max_size = 340
        
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convertir a PhotoImage
        image_pil = Image.fromarray(image_resized)
        photo = ImageTk.PhotoImage(image_pil)
        
        # Mostrar en canvas
        canvas.delete("all")
        x = (canvas.winfo_reqwidth() - new_w) // 2
        y = (canvas.winfo_reqheight() - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Guardar referencia
    
    def display_processed_image(self, image, canvas):
        """Muestra una imagen procesada en un canvas."""
        # Si es escala de grises normalizada, desnormalizar
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        # Si es escala de grises, convertir a RGB para visualización
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para ajustar al canvas
        h, w = image_rgb.shape[:2]
        max_size = 340
        
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # Convertir a PhotoImage
        image_pil = Image.fromarray(image_resized)
        photo = ImageTk.PhotoImage(image_pil)
        
        # Mostrar en canvas
        canvas.delete("all")
        x = (canvas.winfo_reqwidth() - new_w) // 2
        y = (canvas.winfo_reqheight() - new_h) // 2
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        canvas.image = photo  # Guardar referencia
    
    def classify_image(self):
        """Clasifica la imagen cargada."""
        if self.model is None:
            messagebox.showwarning("Advertencia", "Debe cargar un modelo primero")
            return
        
        if self.processed_image is None:
            messagebox.showwarning("Advertencia", "Debe cargar una imagen primero")
            return
        
        try:
            # Extraer características
            features = self.feature_extractor.extract(self.processed_image)
            features = features.reshape(1, -1)
            
            # Predecir
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0]
            
            # Mostrar resultado
            if prediction == 1:
                result_text = "✅ PLACA DETECTADA"
                result_color = "green"
            else:
                result_text = "❌ NO ES UNA PLACA"
                result_color = "red"
            
            self.result_label.config(text=result_text, foreground=result_color)
            self.confidence_label.config(
                text=f"Confianza: {confidence*100:.2f}%",
                foreground="blue"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al clasificar:\n{e}")
    
    def clear_all(self):
        """Limpia todos los campos y resultados."""
        self.image_path = None
        self.original_image = None
        self.processed_image = None
        
        self.original_canvas.delete("all")
        self.processed_canvas.delete("all")
        
        self.result_label.config(text="")
        self.confidence_label.config(text="")


def main():
    """Función principal."""
    root = tk.Tk()
    app = PlateClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
