import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import os

st.title("Análisis de Volúmenes Ex-factory por País")

# Define los archivos Excel de INNOVIX
excel_files = [
    'files/INNOVIX_Elbonie.xlsx',
    'files/INNOVIX_Floresland.xlsx'
]

# Crear figura de Plotly
fig = go.Figure()

# Leer cada archivo Excel y agregar al gráfico
for file in excel_files:
    if os.path.exists(file):
        # Extraer el nombre del país del nombre del archivo
        country = file.split('_')[1].split('.')[0]
        
        # Leer la hoja Ex-factory volumes
        df_ex = pd.read_excel(file, sheet_name='Ex-Factory volumes')
        df_demand = pd.read_excel(file, sheet_name='Demand volumes')
        
        # Filtrar df_demand para solo incluir INNOVIX
        df_demand = df_demand[df_demand['Product'] == 'INNOVIX']
        
        # Convertir fechas a datetime
        df_ex['Date'] = pd.to_datetime(df_ex['Date'])
        df_demand['Date'] = pd.to_datetime(df_demand['Date'])
        
        # Agregar línea de Ex-factory al gráfico
        fig.add_trace(go.Scatter(
            x=df_ex['Date'],
            y=df_ex['Value'],
            name=f'{country} - Ex-factory',
            mode='lines'
        ))
        
        # Agregar línea de Demand al gráfico
        fig.add_trace(go.Scatter(
            x=df_demand['Date'],
            y=df_demand['Value'],
            name=f'{country} - Demand',
            mode='lines',
            line=dict(dash='dash')  # Línea punteada para diferenciar
        ))

# Actualizar el diseño del gráfico
fig.update_layout(
    title='Comparación de Volúmenes Ex-factory y Demand por País',
    xaxis_title='Fecha',
    yaxis_title='Volumen',
    hovermode='x unified'
)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Prueba de Kolmogorov-Smirnov para comparar distribuciones
dfs = []
for file in excel_files:
    if os.path.exists(file):
        df = pd.read_excel(file, sheet_name='Ex-Factory volumes')
        dfs.append(df)

if len(dfs) == 2:
    ks_statistic, p_value = stats.ks_2samp(
        dfs[0]['Value'],
        dfs[1]['Value']
    )
    
    st.subheader("Prueba de Kolmogorov-Smirnov")
    st.write(f"Estadístico KS: {ks_statistic:.4f}")
    st.write(f"Valor p: {p_value:.4f}")
    st.write("Interpretación:")
    if p_value < 0.05:
        st.write("Las distribuciones son significativamente diferentes (p < 0.05)")
    else:
        st.write("No hay evidencia suficiente para decir que las distribuciones son diferentes (p >= 0.05)")
