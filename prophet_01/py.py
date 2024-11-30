import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
nombre_archivo = "ventas_farmaceuticas.csv"
data = pd.read_csv(nombre_archivo)
data.rename(columns={"Fecha": "ds", "Ventas": "y"}, inplace=True)  # Renombrar columnas para Prophet

# Agregar regresores adicionales
regresores = ["Publicidad", "Temperatura", "Eventos"]

# Instanciar el modelo Prophet
model = Prophet()

# Añadir regresores al modelo
for reg in regresores:
    model.add_regressor(reg)

# Ajustar el modelo con los datos
model.fit(data)

# Crear un DataFrame futuro
future = model.make_future_dataframe(periods=90)  # 90 días adicionales
# Agregar valores para los regresores en el futuro
future = future.merge(data[regresores], left_index=True, right_index=True, how='left')  # Copiar datos existentes
future[regresores] = future[regresores].fillna(method='ffill')  # Rellenar faltantes en el futuro

# Hacer predicciones
forecast = model.predict(future)

# Visualizar la predicción
fig = model.plot(forecast)
plt.title("Predicción de Ventas Farmacéuticas con Regresores", fontsize=16)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel("Ventas", fontsize=12)
plt.show()

# Visualizar componentes
fig_components = model.plot_components(forecast)
plt.show()

# Mostrar predicciones finales
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))  # Predicciones finales
