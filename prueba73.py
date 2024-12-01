import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Cargar datos
file_path = 'files/Zegoland - parametrized data (1).xlsx'
train_data = pd.read_excel(file_path, sheet_name='train_data')
train_data['date'] = pd.to_datetime(train_data['date'])

# Crear características temporales
train_data['month'] = train_data['date'].dt.month
train_data['year'] = train_data['date'].dt.year
train_data['trend'] = np.arange(len(train_data))

# Preparar datos
X = train_data.drop(columns=['date', 'truth_sales'])
y = train_data['truth_sales']

# Limpiar NaN y escalar
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_future_features(train_data, periods=3):
    last_date = train_data['date'].max()
    future_dates = pd.date_range(start=last_date, periods=periods+1, freq='ME')[1:]
    
    n_last = 6
    trends = {}
    for col in train_data.drop(columns=['date', 'truth_sales', 'month', 'year', 'trend']):
        trends[col] = train_data[col].tail(n_last).diff().mean()
    
    future_data = pd.DataFrame()
    for col in X.columns:
        if col == 'month':
            future_data[col] = [d.month for d in future_dates]
        elif col == 'year':
            future_data[col] = [d.year for d in future_dates]
        elif col == 'trend':
            future_data[col] = np.arange(len(train_data), len(train_data) + len(future_dates))
        else:
            last_value = train_data[col].iloc[-1]
            future_data[col] = [last_value + (i+1) * trends[col] for i in range(len(future_dates))]
    
    return future_data, future_dates

# Definir el modelo con los mejores hiperparámetros
best_model = GradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    n_estimators=100,
    subsample=0.8,
    random_state=42
)

# Entrenar el modelo con los mejores hiperparámetros
best_model.fit(X_train, y_train)

# Predicción en el conjunto de prueba
y_pred_test = best_model.predict(X_test)

# Calcular métricas de error en el conjunto de prueba
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

print(f"\nMétricas de Error en datos de prueba:")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAPE: {mape_test:.2%}")

# Calcular la media de truth_sales
mean_truth_sales = y.mean()

# Comparar RMSE con la media de truth_sales
print(f"Media de truth_sales: {mean_truth_sales:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"RMSE como porcentaje de la media: {(rmse_test / mean_truth_sales) * 100:.2f}%")

# Crear predicciones futuras
future_data, future_dates = create_future_features(train_data, periods=3)
future_scaled = scaler.transform(future_data)
future_predictions = best_model.predict(future_scaled)

# Añadir ruido aleatorio basado en la desviación estándar histórica
historical_std_dev = np.std(y_train - best_model.predict(X_train))
future_predictions += np.random.normal(0, historical_std_dev, size=future_predictions.shape)

# Visualización
plt.figure(figsize=(12, 6))
plt.plot(train_data['date'], y, label='Ventas Históricas', color='blue', alpha=0.7)
plt.plot(future_dates, future_predictions, 'r--', label='Predicciones Futuras', linewidth=2)

# Conectar datos históricos con predicciones futuras
plt.plot([train_data['date'].iloc[-1], future_dates[0]], [y.iloc[-1], future_predictions[0]], 'r--', linewidth=2)

# Intervalo de confianza
plt.fill_between(future_dates,
                 future_predictions - 1.96*historical_std_dev,
                 future_predictions + 1.96*historical_std_dev,
                 color='red', alpha=0.2)

plt.title('Predicción de Ventas para los Próximos 3 Meses')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()  # Cerrar la figura para evitar que el programa se quede abierto

# Mostrar predicciones y métricas
print("\nPredicciones:")
for date, pred in zip(future_dates, future_predictions):
    print(f"Predicción para {date.strftime('%Y-%m')}: {pred:.2f}")

print(f"\nR² Score en datos de entrenamiento: {best_model.score(X_train, y_train):.3f}")
print(f"R² Score en datos de prueba: {best_model.score(X_test, y_test):.3f}")

print("El programa ha terminado de ejecutarse.")