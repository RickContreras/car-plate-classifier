#!/usr/bin/env python3
"""
Script para comparar modelos HOG y BRISK.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

print("\n" + "="*70)
print(" COMPARACIÓN DE MODELOS HOG vs BRISK")
print("="*70 + "\n")

# Cargar métricas
with open('results/detection_hog_metrics.json', 'r') as f:
    hog_metrics = json.load(f)

with open('results/detection_brisk_metrics.json', 'r') as f:
    brisk_metrics = json.load(f)

# Imprimir comparación
print("📊 MÉTRICAS DE EVALUACIÓN")
print("-" * 70)
print(f"{'Métrica':<30} {'HOG':>15} {'BRISK':>15} {'Mejor':>10}")
print("-" * 70)

metrics = [
    ('IoU Promedio', 'avg_iou'),
    ('IoU Mediana', 'median_iou'),
    ('Desv. Est. IoU', 'std_iou'),
    ('IoU Mínimo', 'min_iou'),
    ('IoU Máximo', 'max_iou'),
    ('MAE', 'mae'),
    ('Precisión @ IoU≥0.5', 'accuracy@0.5'),
    ('Precisión @ IoU≥0.75', 'accuracy@0.75'),
    ('Precisión @ IoU≥0.9', 'accuracy@0.9'),
]

for name, key in metrics:
    hog_val = hog_metrics[key]
    brisk_val = brisk_metrics[key]
    
    # Determinar cuál es mejor (mayor es mejor excepto para MAE y std)
    if key in ['mae', 'std_iou', 'min_iou']:
        mejor = 'HOG' if hog_val <= brisk_val else 'BRISK'
    else:
        mejor = 'HOG' if hog_val >= brisk_val else 'BRISK'
    
    if key.startswith('accuracy'):
        # Mostrar como porcentaje
        print(f"{name:<30} {hog_val*100:>14.2f}% {brisk_val*100:>14.2f}% {mejor:>10}")
    else:
        print(f"{name:<30} {hog_val:>15.4f} {brisk_val:>15.4f} {mejor:>10}")

print("-" * 70)

# Conteo de victorias
hog_wins = sum([
    hog_metrics['avg_iou'] > brisk_metrics['avg_iou'],
    hog_metrics['median_iou'] > brisk_metrics['median_iou'],
    hog_metrics['max_iou'] > brisk_metrics['max_iou'],
    hog_metrics['mae'] < brisk_metrics['mae'],
    hog_metrics['accuracy@0.5'] > brisk_metrics['accuracy@0.5'],
    hog_metrics['accuracy@0.75'] > brisk_metrics['accuracy@0.75'],
    hog_metrics['accuracy@0.9'] > brisk_metrics['accuracy@0.9'],
])

brisk_wins = 7 - hog_wins

print(f"\n🏆 RESUMEN:")
print(f"   HOG gana en:   {hog_wins}/7 métricas principales")
print(f"   BRISK gana en: {brisk_wins}/7 métricas principales")

if hog_wins > brisk_wins:
    print(f"\n✅ GANADOR: HOG (mejor rendimiento general)")
else:
    print(f"\n✅ GANADOR: BRISK (mejor rendimiento general)")

# Información del modelo
print("\n" + "="*70)
print("ℹ️  INFORMACIÓN DE MODELOS")
print("-" * 70)
print(f"{'Característica':<30} {'HOG':>20} {'BRISK':>20}")
print("-" * 70)
print(f"{'Dimensión de features':<30} {42849:>20} {32768:>20}")
print(f"{'Total de muestras':<30} {hog_metrics['total_samples']:>20} {brisk_metrics['total_samples']:>20}")
print(f"{'Correctos @ IoU≥0.5':<30} {hog_metrics['correct@0.5']:>20} {brisk_metrics['correct@0.5']:>20}")
print(f"{'Correctos @ IoU≥0.75':<30} {hog_metrics['correct@0.75']:>20} {brisk_metrics['correct@0.75']:>20}")
print(f"{'Correctos @ IoU≥0.9':<30} {hog_metrics['correct@0.9']:>20} {brisk_metrics['correct@0.9']:>20}")
print("-" * 70)

print("\n" + "="*70)
print("✅ Comparación completada")
print("="*70 + "\n")

# Crear gráfico de comparación
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Comparación de IoU
ax = axes[0, 0]
metrics_names = ['Promedio', 'Mediana', 'Máximo']
hog_values = [hog_metrics['avg_iou'], hog_metrics['median_iou'], hog_metrics['max_iou']]
brisk_values = [brisk_metrics['avg_iou'], brisk_metrics['median_iou'], brisk_metrics['max_iou']]

x = np.arange(len(metrics_names))
width = 0.35

ax.bar(x - width/2, hog_values, width, label='HOG', color='skyblue')
ax.bar(x + width/2, brisk_values, width, label='BRISK', color='lightcoral')
ax.set_ylabel('IoU', fontsize=12)
ax.set_title('Comparación de IoU', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Comparación de Precisión
ax = axes[0, 1]
thresholds = ['IoU≥0.5', 'IoU≥0.75', 'IoU≥0.9']
hog_acc = [hog_metrics['accuracy@0.5']*100, hog_metrics['accuracy@0.75']*100, hog_metrics['accuracy@0.9']*100]
brisk_acc = [brisk_metrics['accuracy@0.5']*100, brisk_metrics['accuracy@0.75']*100, brisk_metrics['accuracy@0.9']*100]

x = np.arange(len(thresholds))
ax.bar(x - width/2, hog_acc, width, label='HOG', color='skyblue')
ax.bar(x + width/2, brisk_acc, width, label='BRISK', color='lightcoral')
ax.set_ylabel('Precisión (%)', fontsize=12)
ax.set_title('Precisión en diferentes umbrales de IoU', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(thresholds)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. MAE Comparison
ax = axes[1, 0]
models = ['HOG', 'BRISK']
mae_values = [hog_metrics['mae'], brisk_metrics['mae']]
colors = ['skyblue', 'lightcoral']

ax.bar(models, mae_values, color=colors, alpha=0.7)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Error Absoluto Medio (MAE)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, v in enumerate(mae_values):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# 4. Distribución de predicciones correctas
ax = axes[1, 1]
categories = ['≥0.5', '≥0.75', '≥0.9']
hog_correct = [hog_metrics['correct@0.5'], hog_metrics['correct@0.75'], hog_metrics['correct@0.9']]
brisk_correct = [brisk_metrics['correct@0.5'], brisk_metrics['correct@0.75'], brisk_metrics['correct@0.9']]

x = np.arange(len(categories))
ax.bar(x - width/2, hog_correct, width, label='HOG', color='skyblue')
ax.bar(x + width/2, brisk_correct, width, label='BRISK', color='lightcoral')
ax.set_ylabel('Número de predicciones correctas', fontsize=12)
ax.set_title('Predicciones correctas por umbral de IoU', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/hog_vs_brisk_comparison.png', dpi=150, bbox_inches='tight')
print("📊 Gráfico de comparación guardado: results/hog_vs_brisk_comparison.png")
