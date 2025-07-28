"""
Métricas de evaluación para modelos de regresión.

Este módulo contiene funciones para calcular diferentes métricas
de rendimiento en problemas de regresión.
"""

def mean_squared_error(y_true, y_pred):
    """
    Calcular el Error Cuadrático Medio (MSE).
    
    Args:
        y_true (list): Valores reales
        y_pred (list): Valores predichos
        
    Returns:
        float: Valor MSE
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Las listas deben tener la misma longitud y no estar vacías")
    mse = 0.0
    i = 0
    while i < len(y_true):
        error = y_true[i] - y_pred[i]
        mse += error ** 2
        i += 1
    mse /= len(y_true)
    return mse

def root_mean_squared_error(y_true, y_pred):
    """
    Calcular la Raíz del Error Cuadrático Medio (RMSE).
    
    Args:
        y_true (list): Valores reales
        y_pred (list): Valores predichos
        
    Returns:
        float: Valor RMSE
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return rmse

def mean_absolute_error(y_true, y_pred):
    """
    Calcular el Error Absoluto Medio (MAE).
    
    Args:
        y_true (list): Valores reales
        y_pred (list): Valores predichos
        
    Returns:
        float: Valor MAE
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Las listas deben tener la misma longitud y no estar vacías")
    mae = 0.0
    i = 0
    while i < len(y_true):
        error = abs(y_true[i] - y_pred[i])
        mae += error
        i += 1
    mae /= len(y_true)
    return mae

def r_squared(y_true, y_pred):
    """
    Calcular el coeficiente de determinación R².
    
    Args:
        y_true (list): Valores reales
        y_pred (list): Valores predichos
        
    Returns:
        float: Valor R² (entre 0 y 1)
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("Las listas deben tener la misma longitud y no estar vacías")
    # Calcular la media de los valores reales
    y_mean = sum(y_true) / len(y_true)
    # Calcular suma de cuadrados total (TSS)
    ss_tot = 0.0
    i = 0
    while i < len(y_true):
        ss_tot += (y_true[i] - y_mean) ** 2
        i += 1
    # Calcular suma de cuadrados residual (RSS)
    ss_res = 0.0
    j = 0
    while j < len(y_true):
        ss_res += (y_true[j] - y_pred[j]) ** 2
        j += 1
    # Calcular R²
    if ss_tot == 0:
        return 1.0
    r2 = 1 - (ss_res / ss_tot)
    return r2
