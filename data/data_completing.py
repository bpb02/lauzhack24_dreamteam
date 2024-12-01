import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Cargar el archivo Excel
archivo_entrada = "Floresland-parametrized.xlsx"  
df = pd.read_excel(archivo_entrada)

# Identificar columnas numéricas
columnas_numericas = df.select_dtypes(include=[np.number]).columns

# Separar las columnas numéricas para rellenar los valores faltantes
datos_numericos = df[columnas_numericas]

# Usar KNN para rellenar valores faltantes
imputer = KNNImputer(n_neighbors=5)  # Ajusta el número de vecinos si es necesario
datos_rellenados = imputer.fit_transform(datos_numericos)

# Reemplazar los datos rellenados en el DataFrame original
df[columnas_numericas] = datos_rellenados

# Guardar el archivo procesado en un nuevo Excel
archivo_salida = "Floresland-filled.xlsx"
df.to_excel(archivo_salida, index=False)

print(f"Archivo procesado guardado como: {archivo_salida}")
