"""
Tests automáticos para las métricas de regresión.
"""

import sys
import os
import unittest
# Ajustar sys.path para importar desde src correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r_squared

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = [2, 4, 6, 8]
        self.y_pred = [2, 4, 6, 8]  # Perfect prediction
        self.y_pred_off = [1, 5, 7, 10]  # Imperfect prediction

    def test_mse_perfect(self):
        self.assertEqual(mean_squared_error(self.y_true, self.y_pred), 0.0)

    def test_rmse_perfect(self):
        self.assertEqual(root_mean_squared_error(self.y_true, self.y_pred), 0.0)

    def test_mae_perfect(self):
        self.assertEqual(mean_absolute_error(self.y_true, self.y_pred), 0.0)

    def test_r2_perfect(self):
        self.assertEqual(r_squared(self.y_true, self.y_pred), 1.0)

    def test_mse_off(self):
        mse = mean_squared_error(self.y_true, self.y_pred_off)
        self.assertTrue(mse > 0)

    def test_rmse_off(self):
        rmse = root_mean_squared_error(self.y_true, self.y_pred_off)
        self.assertTrue(rmse > 0)

    def test_mae_off(self):
        mae = mean_absolute_error(self.y_true, self.y_pred_off)
        self.assertTrue(mae > 0)

    def test_r2_off(self):
        r2 = r_squared(self.y_true, self.y_pred_off)
        self.assertTrue(0 <= r2 < 1)

if __name__ == "__main__":
    unittest.main()
