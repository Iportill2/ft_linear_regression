#!/usr/bin/env python3
"""
Linear Regression Project - Main Entry Point

Este archivo sirve como punto de entrada principal para el proyecto
de regresión lineal.
"""
import sys

def main(args):
    """Función principal del programa."""
    print("=== Linear Regression Project ===")
    print("🚀 Inicializando proyecto...")
    
    # Verificar que se pasó el archivo CSV como argumento
    if len(args) < 2:
        print("❌ Error: Debes especificar un archivo CSV")
        print("💡 Uso: python main.py <archivo_csv>")
        print("📄 Ejemplo: python main.py data/data.csv")
        return
    
    # Importar funciones necesarias
    from src import load_csv_data, split_data, LinearRegression
    from src.plot_utils import plot_regression
    
    print("📊 Cargando datos desde CSV...")
    print(f"📁 Archivo especificado: {args[1]}")
    
    # Cargar datos desde archivo CSV
    X, y = load_csv_data(args[1])
    
    if len(X) > 0 and len(y) > 0:
        print(f"✅ Datos cargados exitosamente: {len(X)} muestras")
        print(f"📊 X (KM)(primeros 5): {X[:5]}")
        print(f"📈 y (€)(primeros 5): {y[:5]}")
        
        # Crear y entrenar modelo
        print("\n🤖 Creando modelo de regresión lineal...")
        model = LinearRegression()
        print("🔧 Entrenando modelo...")
        model.fit(X, y)
        # Comprobación de pendiente descendente
        if model.slope is not None and model.slope >= 0:
            print("\033[91m⚠️  Advertencia: La pendiente del modelo no es descendente (m >= 0).\033[0m")
        else:
            print("\033[92m✅ La pendiente del modelo es descendente (m < 0).\033[0m")
        # Hacer predicciones sobre todos los datos
        predictions = model.predict(X)
        print("✅ Pipeline completo ejecutado!")
        # Guardar gráfico de regresión con el nombre del CSV
        import os
        csv_filename = os.path.basename(args[1])
        png_filename = os.path.splitext(csv_filename)[0] + ".png"
        plot_regression(X, y, predictions, filename=png_filename)
        print(f"✅ Gráfico guardado en graphics/{png_filename}")
        # Abrir el archivo PNG generado con el visor predeterminado
        try:
            import subprocess
            subprocess.run(["xdg-open", os.path.join("graphics", png_filename)], check=False)
        except Exception as e:
            print(f"⚠️  No se pudo abrir el archivo automáticamente: {e}")
        
    else:
        print("❌ No se pudieron cargar los datos del CSV")
    
    # Mostrar estructura del proyecto
    #print("\n📁 Estructura del proyecto:")
    #print("├── data/       - Datasets y archivos de datos")
    #print("├── src/        - Código fuente principal")
    #print("├── notebooks/  - Jupyter notebooks")
    #print("├── tests/      - Tests unitarios")
    #print("└── .venv/      - Entorno virtual Python")


if __name__ == "__main__":
    main(sys.argv)
