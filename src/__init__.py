"""
Linear Regression Package

Módulo principal para implementación de regresión lineal desde cero.
"""

from .linear_regression import LinearRegression
from .metrics import (
    mean_squared_error,
    root_mean_squared_error, 
    mean_absolute_error,
    r_squared
)
from .data_utils import (
    generate_linear_data,
    load_csv_data,
    split_data
)

__version__ = "1.0.0"
__author__ = "Iker"

__all__ = [
    "LinearRegression",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error", 
    "r_squared",
    "generate_linear_data",
    "load_csv_data",
    "split_data"
]
