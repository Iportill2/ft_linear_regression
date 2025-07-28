"""
Linear Regression Implementation

Este módulo contiene la implementación principal del algoritmo
de regresión lineal desde cero.
"""

class LinearRegression:
    """
    Implementación de Regresión Lineal Simple usando Mínimos Cuadrados.
    
    Attributes:
        slope (float): Pendiente de la línea (m)
        intercept (float): Intersección con el eje Y (b)
        is_fitted (bool): Indica si el modelo ha sido entrenado
    """
    
    def __init__(self):
        """Inicializar el modelo de regresión lineal."""
        self.slope = None
        self.intercept = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Entrenar el modelo con los datos de entrada.
        
        Args:
            X (list or array): Variables independientes
            y (list or array): Variables dependientes
        """
        # Validar entrada
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        if len(X) < 2:
            raise ValueError("Se necesitan al menos 2 puntos de datos para entrenar")
        
        print("🔧 Entrenando modelo con algoritmo de mínimos cuadrados...")
        
        # Convertir a listas si no lo son
        X = list(X)
        y = list(y)
        
        # Calcular estadísticas necesarias
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(y)
        sum_xy = 0
        i = 0
        while i < len(X):
            sum_xy += X[i] * y[i]
            i += 1
        sum_x_squared = 0
        j = 0
        while j < len(X):
            sum_x_squared += X[j] * X[j]
            j += 1
        
        # Mostrar estadísticas de cálculo
        print(f"   Número de muestras: {n}")
        print(f"   🧮 Calculando parámetros usando mínimos cuadrados...")
        
        # Calcular denominador para la pendiente
        denominator = n * sum_x_squared - sum_x * sum_x
        
        # Verificar que no hay división por cero (todos los X son iguales)
        if denominator == 0:
            raise ValueError("Todos los valores X son iguales. No se puede ajustar una línea.")
        
        # Calcular pendiente (slope) usando la fórmula de mínimos cuadrados
        # m = (n*Σ(xy) - Σ(x)*Σ(y)) / (n*Σ(x²) - (Σ(x))²)
        self.slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Calcular intersección (intercept) 
        # b = (Σ(y) - m*Σ(x)) / n
        self.intercept = (sum_y - self.slope * sum_x) / n
        
        # Marcar como entrenado
        self.is_fitted = True
        
        # Mostrar resultados
        print(f"   📈 Pendiente (m): {self.slope:.4f}")
        print(f"   📍 Intersección (b): {self.intercept:.4f}")
        print(f"   📝 Ecuación: y = {self.slope:.4f}x + {self.intercept:.4f}")
        print("✅ Modelo entrenado exitosamente!")
        
    def predict(self, X):
        """
        Realizar predicciones con el modelo entrenado.
        
        Args:
            X (list or array): Valores para predecir
            
        Returns:
            list: Predicciones del modelo
        """
        if self.is_fitted == False:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Validar entrada
        if len(X) == 0:
            print("⚠️  No hay valores para predecir")
            return []
        
        # Convertir a lista si no lo es
        X = list(X)
        
        print(f"📈 Realizando predicciones para {len(X)} valores...")
        print(f"   📝 Usando ecuación: y = {self.slope:.4f}x + {self.intercept:.4f}")
        
        # Aplicar la ecuación lineal: y = mx + b
        predictions = []
        for x_val in X:
            y_pred = self.slope * x_val + self.intercept
            predictions.append(y_pred)
        
        # Mostrar estadísticas de las predicciones
        print(f"   📊 Predicciones generadas:")
        print(f"   🔍 Rango de entrada X: [{min(X):.2f}, {max(X):.2f}]")
        print(f"   📈 Rango de predicciones: [{min(predictions):.2f}, {max(predictions):.2f}]")
        
        # Mostrar algunos ejemplos si hay pocos valores
        if len(X) <= 5:
            print("   💡 Ejemplos de predicciones:")
            for i, (x_val, y_pred) in enumerate(zip(X, predictions)):
                print(f"      x = {x_val:.2f} → y = {y_pred:.2f}")
        
        print(f"✅ {len(predictions)} predicciones completadas!")
        
        return predictions
    
    def mse(self, X, y):
        """
        Calcular el Error Cuadrático Medio (Mean Squared Error).
        
        Args:
            X (list or array): Variables independientes
            y (list or array): Variables dependientes reales
            
        Returns:
            float: Valor MSE del modelo
        """
        if  self.is_fitted == False:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Validar entrada
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        if len(X) == 0:
            print("⚠️  No hay datos para evaluar")
            return float('inf')
        
        print(f"📊 Calculando MSE para {len(X)} puntos...")
        
        # Hacer predicciones (sin mostrar info detallada)
        X = list(X)
        y = list(y)
        
        # Calcular predicciones sin prints
        y_pred = []
        i = 0
        while i < len(X):
            pred = self.slope * X[i] + self.intercept
            y_pred.append(pred)
            i += 1
        
        # Calcular errores cuadrados
        errores_cuadrados = []
        j = 0
        while j < len(y):
            error = y[j] - y_pred[j]
            error_cuadrado = error ** 2
            errores_cuadrados.append(error_cuadrado)
            j += 1
        
        # Calcular MSE
        mse_value = sum(errores_cuadrados) / len(errores_cuadrados)
        
        # Mostrar estadísticas
        # Calcular errores individuales para estadísticas
        errores = []
        k = 0
        while k < len(y):
            error = y[k] - y_pred[k]
            errores.append(error)
            k += 1
        
        print(f"   📈 MSE: {mse_value:.4f}")
        print(f"   📊 Error promedio: ±{(mse_value ** 0.5):.4f}")
        print(f"   🔍 Rango de errores: [{min(errores):.4f}, {max(errores):.4f}]")
        
        # Interpretación
        if mse_value < 0.1:
            print("   ✅ ¡Excelente precisión!")
        elif mse_value < 1.0:
            print("   👍 Buena precisión")
        elif mse_value < 10.0:
            print("   ⚠️  Precisión moderada")
        else:
            print("   ❌ Baja precisión - modelo necesita mejoras")
        
        return mse_value
    
    def score(self, X, y):
        """
        Calcular el coeficiente de determinación R².
        
        Args:
            X (list or array): Variables independientes
            y (list or array): Variables dependientes reales
            
        Returns:
            float: Valor R² del modelo (entre 0 y 1, donde 1 es perfecto)
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Validar entrada
        if len(X) != len(y):
            raise ValueError("X e y deben tener la misma longitud")
        
        if len(X) == 0:
            print("⚠️  No hay datos para evaluar")
            return 0.0
        
        print(f"📊 Calculando R² para {len(X)} puntos...")
        
        # Convertir a listas
        X = list(X)
        y = list(y)
        
        # Calcular predicciones sin prints
        y_pred = []
        i = 0
        while i < len(X):
            pred = self.slope * X[i] + self.intercept
            y_pred.append(pred)
            i += 1
        
        # Calcular media de y reales
        y_mean = sum(y) / len(y)
        
        # Calcular suma de cuadrados total (TSS)
        ss_tot = sum((real - y_mean) ** 2 for real in y)
        
        # Calcular suma de cuadrados residual (RSS)
        ss_res = 0
        j = 0
        while j < len(y):
            ss_res += (y[j] - y_pred[j]) ** 2
            j += 1
        
        # Calcular R²
        if ss_tot == 0:
            r2 = 1.0  # Caso especial: todos los y son iguales
        else:
            r2 = 1 - (ss_res / ss_tot)
        
        # Mostrar estadísticas
        print(f"   📈 R²: {r2:.4f}")
        print(f"   📊 Porcentaje explicado: {r2 * 100:.2f}%")
        
        # Interpretación
        if r2 >= 0.9:
            print("   ✅ ¡Excelente ajuste!")
        elif r2 >= 0.7:
            print("   👍 Buen ajuste")
        elif r2 >= 0.5:
            print("   ⚠️  Ajuste moderado")
        elif r2 >= 0.0:
            print("   ❌ Ajuste pobre")
        else:
            print("   💥 Modelo peor que una línea horizontal")
        
        return r2
