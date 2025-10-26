#!/bin/bash

# Script para ejecutar la aplicación web del clasificador de placas

echo "🚀 Iniciando aplicación web del Clasificador de Placas..."
echo ""

# Activar entorno virtual
source venv/bin/activate

# Ejecutar aplicación
echo "📡 Servidor iniciando en http://localhost:7860"
echo "🌐 Para acceder desde otra computadora, usa: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "💡 Presiona Ctrl+C para detener el servidor"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python app/web_app.py
