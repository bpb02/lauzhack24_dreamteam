import pandas as pd
import numpy as np

def crear_csv_ventas(nombre_archivo):
    np.random.seed(42)  # Para reproducibilidad
    fechas = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")  # Fechas de 2 años
    ventas = np.random.randint(50, 200, size=len(fechas)) + np.sin(np.linspace(0, 10, len(fechas))) * 10  # Ventas simuladas
    
    # Variables adicionales
    publicidad = np.random.randint(500, 1500, size=len(fechas))  # Gasto en publicidad
    temperatura = 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, len(fechas)))  # Oscilación estacional de temperatura
    eventos = np.random.choice([0, 1], size=len(fechas), p=[0.9, 0.1])  # Eventos especiales (0 o 1)
    
    # Crear el DataFrame
    datos = pd.DataFrame({
        "Fecha": fechas,
        "Ventas": ventas,
        "Publicidad": publicidad,
        "Temperatura": temperatura,
        "Eventos": eventos
    })
    datos.to_csv(nombre_archivo, index=False)
    print(f"Archivo CSV '{nombre_archivo}' creado con éxito.")

# Llamada a la función
if __name__ == "__main__":
    nombre_archivo = "ventas_farmaceuticas.csv"
    crear_csv_ventas(nombre_archivo)
