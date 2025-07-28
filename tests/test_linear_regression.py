import os
CSV_FILENAME = os.environ.get("CSV_TEST_FILE", "data/data.csv")

def load_csv_data(csv_file=CSV_FILENAME):
    import csv
    X, y = [], []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                X.append(float(row[0]))
                y.append(float(row[1]))
    return X, y


passed_tests = 0
total_tests = 0

def print_summary():
    if total_tests > 0:
        percent = (passed_tests / total_tests) * 100
        print(f"\n============================================= {passed_tests} passed of {total_tests} ({percent:.0f}%) ==============================================")

def print_summary():
    if total_tests > 0:
        percent = (passed_tests / total_tests) * 100
        print(f"\n============================================= {passed_tests} passed of {total_tests} ({percent:.0f}%) ==============================================")


import pytest
import sys
import os
# Añadir src al path para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
src_dir = os.path.join(parent_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from linear_regression import LinearRegression
from metrics import mean_squared_error, r_squared
#from data_utils import generate_linear_data



class TestLinearRegression:
    def test_init(self):
        model = LinearRegression()
        assert model.slope is None
        assert model.intercept is None
        assert model.is_fitted is False

    @pytest.mark.parametrize("csv_file", [CSV_FILENAME])
    def test_fit_basic(self, csv_file):
        model = LinearRegression()
        X, y = load_csv_data(csv_file)
        model.fit(X, y)
        assert model.is_fitted is True

    def test_predict_without_fit(self):
        model = LinearRegression()
        with pytest.raises(ValueError):
            model.predict([1, 2, 3])



class TestMetrics:
    @pytest.mark.parametrize("csv_file", [CSV_FILENAME])
    def test_mse_placeholder(self, csv_file):
        X, y = load_csv_data(csv_file)
        y_true, y_pred = y, y
        result = mean_squared_error(y_true, y_pred)
        if isinstance(result, float):
            print(f"✅ MSE: {result:.6f}")
        else:
            print(f"❌ MSE: {result}")



class TestDataUtils:
    @pytest.mark.parametrize("csv_file", [CSV_FILENAME])
    def test_generate_linear_data(self, csv_file):
        import math
        from linear_regression import LinearRegression
        from metrics import r_squared
        X, y = load_csv_data(csv_file)
        print(f"🔢 Datos cargados: {len(X)} muestras")
        # Comprobar tipo
        assert isinstance(X, list), "❌ X debe ser una lista"
        assert isinstance(y, list), "❌ y debe ser una lista"
        print("✅ Tipos correctos")
        # Comprobar longitud
        n_samples = len(X)
        assert len(X) == n_samples, f"❌ X debe tener {n_samples} elementos"
        assert len(y) == n_samples, f"❌ y debe tener {n_samples} elementos"
        print("✅ Longitud correcta")
        # Comprobar valores numéricos y ausencia de NaN/infinito
        i = 0
        while i < len(X):
            x = X[i]
            assert isinstance(x, (int, float)), f"❌ X[{i}] no es numérico"
            assert not math.isnan(x), f"❌ X[{i}] es NaN"
            assert not math.isinf(x), f"❌ X[{i}] es infinito"
            i += 1
        print("✅ X sin NaN/infinito")
        j = 0
        while j < len(y):
            val = y[j]
            assert isinstance(val, (int, float)), f"❌ y[{j}] no es numérico"
            assert not math.isnan(val), f"❌ y[{j}] es NaN"
            assert not math.isinf(val), f"❌ y[{j}] es infinito"
            j += 1
        print("✅ y sin NaN/infinito")
        # Comprobar que hay variación en los datos
        assert len(set(X)) > 1, "❌ X debe contener valores distintos"
        assert len(set(y)) > 1, "❌ y debe contener valores distintos"
        print("✅ Variación confirmada")
        # Comprobar rango de valores
        assert min(X) < max(X), "❌ X debe tener un rango válido"
        assert min(y) < max(y), "❌ y debe tener un rango válido"
        print(f"✅ Rango de X: [{min(X)}, {max(X)}], Rango de y: [{min(y)}, {max(y)}]")
        # Comprobar correlación lineal aproximada
        def pearson_corr(a, b):
            mean_a = sum(a) / len(a)
            mean_b = sum(b) / len(b)
            num = 0.0
            k = 0
            while k < len(a):
                num += (a[k] - mean_a) * (b[k] - mean_b)
                k += 1
            den_a = 0.0
            l = 0
            while l < len(a):
                den_a += (a[l] - mean_a) ** 2
                l += 1
            den_a = math.sqrt(den_a)
            den_b = 0.0
            m = 0
            while m < len(b):
                den_b += (b[m] - mean_b) ** 2
                m += 1
            den_b = math.sqrt(den_b)
            if den_a and den_b:
                return num / (den_a * den_b)
            else:
                return 0.0
        corr = pearson_corr(X, y)
        print(f"🔗 Correlación calculada: {corr:.4f}")
        assert abs(corr) > 0.8, f"❌ La correlación lineal entre X e y debe ser alta (|corr| > 0.8), obtenida: {corr}"
        print("✅ Correlación alta")
        # Comprobar que se puede ajustar un modelo lineal con buen R²
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r_squared(y, y_pred)
        print(f"📊 R² obtenido: {r2:.4f}")
        assert r2 > 0.7, f"❌ El modelo ajustado debe tener un R² aceptable (>0.7), obtenido: {r2}"
        print("✅ R² aceptable")
        print("🎉 [Test] test_generate_linear_data completado exitosamente!")
