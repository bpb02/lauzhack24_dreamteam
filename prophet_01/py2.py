import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(excel_path):
    """
    Carga los datos del archivo Excel y prepara el DataFrame para Prophet
    
    Parámetros:
    excel_path (str): Ruta al archivo Excel
    
    Retorna:
    tuple: DataFrame de datos y DataFrame de diccionario
    """
    # Cargar datos de Excel
    data_df = pd.read_excel(excel_path, sheet_name='data')
    dict_df = pd.read_excel(excel_path, sheet_name='dictionary')
    
    # Preparar DataFrame para Prophet
    prophet_df = data_df[['date', 'truth']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    return prophet_df, data_df, dict_df

def preprocess_data(data_df, prophet_df):
    """
    Preprocesa los datos para manejar valores NaN y preparar regresores
    
    Parámetros:
    data_df (DataFrame): DataFrame original
    prophet_df (DataFrame): DataFrame de Prophet
    
    Retorna:
    tuple: DataFrame procesado y lista de regresores
    """
    # Identificar columnas numéricas potenciales como regresores
    potential_regressors = [col for col in data_df.columns 
                             if col not in ['date', 'truth'] and data_df[col].dtype in ['int64', 'float64']]
    
    # Crear una copia del DataFrame de Prophet para modificar
    processed_df = prophet_df.copy()
    
    # Manejar NaNs en regresores
    for regressor in potential_regressors:
        # Si hay NaNs, reemplazar con la mediana o media
        if data_df[regressor].isnull().any():
            print(f"Advertencia: Encontrados NaNs en {regressor}. Rellenando con mediana.")
            fill_value = data_df[regressor].median()
            processed_df[regressor] = data_df[regressor].fillna(fill_value)
        else:
            processed_df[regressor] = data_df[regressor]
    
    return processed_df, potential_regressors

def train_prophet_model(df, additional_regressors=None):
    """
    Entrena un modelo Prophet con posibles regresores adicionales
    
    Parámetros:
    df (DataFrame): DataFrame preparado para Prophet
    additional_regressors (list, opcional): Lista de columnas adicionales como regresores
    
    Retorna:
    tuple: Modelo Prophet entrenado y modelo ajustado
    """
    # Inicializar modelo Prophet
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # Añadir regresores adicionales si se proporcionan
    if additional_regressors:
        for regressor in additional_regressors:
            model.add_regressor(regressor)
    
    # Entrenar modelo
    model.fit(df)
    
    return model

def evaluate_model_performance(model, df):
    """
    Evalúa el rendimiento del modelo mediante validación cruzada
    
    Parámetros:
    model (Prophet): Modelo Prophet entrenado
    df (DataFrame): DataFrame original
    
    Retorna:
    DataFrame con métricas de rendimiento
    """
    # Realizar validación cruzada
    df_cv = cross_validation(
        model, 
        initial='365 days', 
        period='180 days', 
        horizon='90 days'
    )
    
    # Calcular métricas de rendimiento
    df_performance = performance_metrics(df_cv)
    
    return df_performance

def visualize_results(model, forecast, performance_metrics):
    """
    Genera visualizaciones de los resultados del modelo
    
    Parámetros:
    model (Prophet): Modelo Prophet
    forecast (DataFrame): Predicciones del modelo
    performance_metrics (DataFrame): Métricas de rendimiento
    """
    try:
        # Gráfico de predicciones
        fig1 = model.plot(forecast)
        plt.title('Predicción de Ventas')
        plt.tight_layout()
        plt.savefig('sales_forecast.png')
        plt.close()
        
        # Gráfico de componentes
        fig2 = model.plot_components(forecast)
        plt.title('Componentes de la Predicción')
        plt.tight_layout()
        plt.savefig('forecast_components.png')
        plt.close()
        
        # Visualizar métricas de rendimiento
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=performance_metrics)
        plt.title('Métricas de Rendimiento del Modelo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.close()
    except Exception as e:
        print(f"Error al generar visualizaciones: {e}")
   
def advanced_visualization(prophet_df, forecast, performance):
    """
    Genera visualizaciones avanzadas para análisis de predicción de ventas
    """
    plt.figure(figsize=(15, 10))
    
    # 1. Serie temporal original vs predicción
    plt.subplot(2, 2, 1)
    plt.plot(prophet_df['ds'], prophet_df['y'], label='Datos Originales')
    plt.plot(forecast['ds'], forecast['yhat'], color='red', label='Predicción')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='pink', alpha=0.3, label='Intervalo de Confianza')
    plt.title('Ventas Reales vs Predicción')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 2. Error de predicción
    plt.subplot(2, 2, 2)
    error = np.abs(prophet_df['y'] - forecast['yhat'][:len(prophet_df)])
    plt.plot(prophet_df['ds'], error)
    plt.title('Error Absoluto de Predicción')
    plt.xticks(rotation=45)
    
    # 3. Distribución de métricas de rendimiento
    plt.subplot(2, 2, 3)
    performance.boxplot()
    plt.title('Métricas de Rendimiento')
    plt.xticks(rotation=45)
    
    # 4. Residuos del modelo
    plt.subplot(2, 2, 4)
    residuals = prophet_df['y'] - forecast['yhat'][:len(prophet_df)]
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title('Distribución de Residuos')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png')
    plt.close()

    
    # Guardar predicciones
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('sales_forecast.csv', index=False)
    performance.to_csv('performance_metrics.csv', index=False)

def main(excel_path):
    """
    Función principal para ejecutar todo el proceso de predicción
    
    Parámetros:
    excel_path (str): Ruta al archivo Excel
    """
    # Cargar datos
    prophet_df, data_df, dict_df = load_and_prepare_data(excel_path)
    
    # Preprocesar datos y obtener regresores
    processed_df, potential_regressors = preprocess_data(data_df, prophet_df)
    
    # Entrenar modelo
    model = train_prophet_model(
        processed_df, 
        additional_regressors=potential_regressors
    )
    
    # Generar predicciones
    future = model.make_future_dataframe(periods=90)  # Predecir 90 días
    
    # Si hay regresores, añadirlos al futuro DataFrame
    if potential_regressors:
        for regressor in potential_regressors:
            future[regressor] = np.mean(processed_df[regressor])
    
    forecast = model.predict(future)
    
    # Evaluar rendimiento
    performance = evaluate_model_performance(model, processed_df)
    
    # Calcular métricas de precisión
    mape = np.mean(np.abs((processed_df['y'] - forecast['yhat'][:len(processed_df)]) / processed_df['y'])) * 100
    rmse = np.sqrt(np.mean((processed_df['y'] - forecast['yhat'][:len(processed_df)])**2))
    
    print("Métricas de Precisión:")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    
    # Visualizar resultados
    visualize_results(model, forecast, performance)
    advanced_visualization(processed_df, forecast, performance)
    
if __name__ == "__main__":
    main('Zegoland - parametrized data.xlsx')