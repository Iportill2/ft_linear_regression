"""
Funciones de visualización para regresión lineal.
"""

import matplotlib.pyplot as plt
import os

def plot_regression(X, y, y_pred, filename):
 
    #plt.figure(figsize=(8, 5), facecolor="grey")
    #plt.figure(figsize=(8, 5), facecolor=(0.5, 0.5, 0.5))   
    plt.figure(figsize=(8, 5), facecolor=("#e5d48f"))  # Ajuste del color de fondo
    plt.gca().set_facecolor("#d3d3d395")  # Ajuste de color para la gráfica

    plt.scatter(X, y, color="#089232", label="Datos reales")
    plt.plot(X, y_pred, color="#ff4c00", label="Recta de regresión")
    plt.xlabel("Kms recorridos")
    plt.ylabel("Precio (€)")
    plt.title("Regresión Lineal en Precio de Vehículos")
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join("graphics", filename)
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Gráfico guardado como {filename}")
