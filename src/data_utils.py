"""
Utilidades para manejo y generación de datos.

Este módulo contiene funciones auxiliares para cargar, generar
y manipular datasets para el proyecto.
"""

def generate_linear_data(n_samples=100, noise=0.1, slope=2.0, intercept=1.0):
    """
    Generar datos sintéticos con relación lineal.
    
    Args:
        n_samples (int): Número de muestras a generar
        noise (float): Nivel de ruido en los datos
        slope (float): Pendiente real de la línea
        intercept (float): Intersección real con el eje Y
        
    Returns:
        tuple: (X, y) datos generados
    """
    import random
    
    print(f"📊 Generando {n_samples} muestras sintéticas...")
    print(f"   📈 Pendiente: {slope}")
    print(f"   📍 Intersección: {intercept}")
    print(f"   🔊 Nivel de ruido: {noise}")
    
    X = []
    y = []
    
    # Generar valores X distribuidos uniformemente
    for i in range(n_samples):
        # X entre 0 y 100 (puedes ajustar el rango según necesites)
        x_val = i * (100.0 / n_samples) + random.uniform(-5, 5)
        
        # Calcular y usando la ecuación lineal: y = slope * x + intercept
        y_perfect = slope * x_val + intercept
        
        # Añadir ruido aleatorio
        noise_value = random.uniform(-noise, noise) * abs(y_perfect)
        y_val = y_perfect + noise_value
        
        X.append(x_val)
        y.append(y_val)
    
    print(f"✅ Datos generados exitosamente")
    print(f"   📊 Rango X: [{min(X):.2f}, {max(X):.2f}]")
    print(f"   📈 Rango y: [{min(y):.2f}, {max(y):.2f}]")
    
    return X, y

def load_csv_data(filepath):
    """
    Cargar datos desde un archivo CSV.
    
    Args:
        filepath (str): Ruta al archivo CSV
        
    Returns:
        tuple: (X, y) datos cargados
    """
    X = []
    y = []

    try:
        print(f"📁 Cargando datos desde {filepath}...")
        with open(filepath, 'r') as file:
            lines = file.readlines()
            
            # Saltar la primera línea (headers)
            first_data_line = True
            skipped_lines = 0
            processed_lines = 0
            
            for line_num, line in enumerate(lines[1:], start=2):  # Empezar desde línea 2
                # Saltar líneas vacías
                if not line.strip():
                    continue
                    
                # Dividir por coma y limpiar espacios
                values = [val.strip() for val in line.strip().split(',')]
                
                # Advertir sobre columnas extra solo una vez
                if first_data_line and len(values) > 2:
                    print(f"\033[91mℹ️  Advertencia: CSV tiene {len(values)} columnas, usando solo las primeras 2\033[0m")
                    first_data_line = False
                
                # Verificar número de columnas
                if len(values) < 2:
                    print(f"⚠️  Línea {line_num}: Solo {len(values)} valor(es) encontrado(s), se necesitan al menos 2. Saltando línea.")
                    skipped_lines += 1
                    continue
                elif len(values) == 1:
                    print(f"⚠️  Línea {line_num}: Solo 1 valor encontrado: '{values[0]}'. Saltando línea.")
                    skipped_lines += 1
                    continue
                
                try:
                    # Primera columna = X (variable independiente)
                    x_val = float(values[0])
                    
                    # Segunda columna = y (variable dependiente)  
                    y_val = float(values[1])
                    
                    X.append(x_val)
                    y.append(y_val)
                    processed_lines += 1
                    
                except ValueError as ve:
                    print(f"⚠️  Línea {line_num}: Error de conversión a número: '{values[0]}', '{values[1]}' - {ve}. Saltando línea.")
                    skipped_lines += 1
                    continue
                    
        # Mostrar resumen de carga
        total_data_lines = len(lines) - 1  # Excluir header
        
        # Validar si el formato es adecuado - MODO ESTRICTO
        if skipped_lines > 0:
            print(f"❌ ERROR: Formato de datos inadecuado")
            print(f"   📋 Se encontraron {skipped_lines} línea(s) con errores de formato")
            print(f"   📊 Líneas totales procesadas: {total_data_lines}")
            print(f"   ✅ Líneas válidas encontradas: {processed_lines}")
            print(f"   🚫 Líneas con errores: {skipped_lines}")
            print(f"   💡 Formato requerido: Todas las líneas deben tener exactamente 2 números")
            print(f"   📝 Ejemplo correcto: 150000,4000")
            return [], []
        
        if processed_lines == 0:
            print(f"❌ ERROR: No se encontraron datos válidos")
            print(f"   � El archivo debe contener al menos una línea con 2 números")
            print(f"   💡 Formato esperado: numero1,numero2")
            return [], []
        
        print(f"✅ Procesadas {processed_lines} líneas correctamente")
        print(f"✅ Formato de archivo correcto - sin errores detectados")
        print(f"📊 Total de datos cargados: {len(X)} puntos")
        return X, y
        
    except FileNotFoundError:
        print(f"❌ Error: El archivo {filepath} no existe.")
        return [], []
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return [], []

def split_data(X, y, test_size=0.2):
    """
    Dividir datos en conjuntos de entrenamiento y prueba.
    
    Args:
        X (list): Variables independientes
        y (list): Variables dependientes
        test_size (float): Proporción para el conjunto de prueba
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    import random
    
    # Validar entrada
    if len(X) != len(y):
        print("❌ Error: X e y deben tener la misma longitud")
        return [], [], [], []
    
    if len(X) == 0:
        print("❌ Error: No hay datos para dividir")
        return [], [], [], []
    
    if test_size < 0 or test_size > 1:
        print("❌ Error: test_size debe estar entre 0 y 1")
        return [], [], [], []
    
    print(f"🔀 Dividiendo datos: {int((1-test_size)*100)}% entrenamiento, {int(test_size*100)}% prueba")
    
    # Crear índices para mezclar aleatoriamente
    indices = list(range(len(X)))
    random.shuffle(indices)  # Mezclar aleatoriamente
    
    # Calcular tamaños
    total_samples = len(X)
    test_samples = int(total_samples * test_size)
    train_samples = total_samples - test_samples
    
    # Dividir índices
    train_indices = indices[:train_samples]
    test_indices = indices[train_samples:]
    
    # Crear conjuntos usando los índices
    X_train = [X[i] for i in train_indices]
    y_train = [y[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_test = [y[i] for i in test_indices]
    
    # Mostrar estadísticas
    print(f"   📊 Total de muestras: {total_samples}")
    print(f"   🏋️  Entrenamiento: {len(X_train)} muestras")
    print(f"   🧪 Prueba: {len(X_test)} muestras")
    print(f"   🎲 Datos mezclados aleatoriamente")
    
    # Mostrar rangos de los conjuntos
    if len(X_train) > 0:
        print(f"   📈 Rango X entrenamiento: [{min(X_train):.2f}, {max(X_train):.2f}]")
        print(f"   📊 Rango y entrenamiento: [{min(y_train):.2f}, {max(y_train):.2f}]")
    
    if len(X_test) > 0:
        print(f"   🔍 Rango X prueba: [{min(X_test):.2f}, {max(X_test):.2f}]")
        print(f"   📋 Rango y prueba: [{min(y_test):.2f}, {max(y_test):.2f}]")
    
    return X_train, X_test, y_train, y_test
