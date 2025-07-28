# Linear Regression Project

## 📋 Descripción
Implementación de regresión lineal desde cero en Python, desarrollando algoritmos de machine learning fundamentales.

## 🎯 Objetivos
- Implementar regresión lineal simple y múltiple desde cero
- Entender la matemática detrás del algoritmo
- Crear métricas de evaluación (MSE, RMSE, R²)
- Visualizar resultados y análisis de datos

## 📁 Estructura del Proyecto
```
linear_regression/
├── .venv/                 # Entorno virtual Python
├── data/                  # Datasets y archivos de datos
├── src/                   # Código fuente principal
├── notebooks/             # Jupyter notebooks para análisis
├── tests/                 # Tests unitarios
├── requirements.txt       # Dependencias del proyecto
├── README.md             # Este archivo
└── main.py               # Punto de entrada principal
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.10+
- pip

### Configuración del entorno
```bash
# Clonar el repositorio
git clone <repository-url>
cd linear_regression

# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso
```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar programa principal
python main.py
```
## 🧪 Tests parametrizables por archivo de datos

Puedes elegir el archivo CSV que se usará en los tests sin modificar el código, usando la variable de entorno `CSV_TEST_FILE`.

Por ejemplo, para ejecutar los tests con un archivo específico:

```bash
source .venv/bin/activate
CSV_TEST_FILE=data/data.csv pytest tests/test_linear_regression.py -v
```

```bash
source .venv/bin/activate
CSV_TEST_FILE=data/data.csv pytest -v
```
# Para ejecutar el programa
```bash
source .venv/bin/activate
CSV_TEST_FILE=data/otro_archivo.csv python main.py data/data.csv
```



Si no defines la variable, se usará por defecto `data/data.csv`.

Esto es posible porque en `tests/test_linear_regression.py` se define:

```python
import os
CSV_FILENAME = os.environ.get("CSV_TEST_FILE", "data/data.csv")
```

Todos los tests usan esa variable global para cargar el archivo de datos.

## 🧮 Algoritmo Implementado

### Regresión Lineal Simple
- **Ecuación**: y = mx + b
- **Método**: Mínimos cuadrados (Least Squares)
- **Optimización**: Gradiente descendente

### Métricas de Evaluación
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)
- **MAE** (Mean Absolute Error)

## 📊 Datasets
- Datos sintéticos generados programáticamente
- Datasets públicos para testing
- Datos de ejemplo incluidos en `/data/`

## 🔧 Desarrollo
```bash
# Ejecutar tests
python -m pytest tests/

# Linter de código
flake8 src/

# Formatear código
black src/
```

## 📈 Resultados Esperados
- Predicciones precisas en datos de prueba
- Visualizaciones claras de ajuste del modelo
- Métricas de rendimiento documentadas



## 👨‍💻 Autor
**Iker Portillo** - Proyecto de Machine Learning

---
*Proyecto desarrollado como parte del aprendizaje de algoritmos de Machine Learning*
