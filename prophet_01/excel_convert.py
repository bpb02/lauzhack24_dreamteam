import pandas as pd
import os

def convert_excel_to_csv(excel_path, output_folder=None):
    """
    Convierte todas las hojas de un archivo Excel a archivos CSV.
    
    Parámetros:
    - excel_path: Ruta completa al archivo Excel
    - output_folder: Carpeta de salida para los CSVs (opcional)
    """
    # Si no se especifica carpeta de salida, usar la misma del archivo Excel
    if output_folder is None:
        output_folder = os.path.dirname(excel_path)
    
    # Nombre base del archivo sin extensión
    base_filename = os.path.splitext(os.path.basename(excel_path))[0]
    
    # Leer el archivo Excel
    xl = pd.ExcelFile(excel_path)
    
    # Iterar sobre todas las hojas
    for sheet_name in xl.sheet_names:
        # Leer la hoja
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how='all')
        
        # Crear nombre de archivo CSV
        csv_filename = f"{base_filename}_{sheet_name}.csv"
        csv_path = os.path.join(output_folder, csv_filename)
        
        # Guardar como CSV
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Hoja '{sheet_name}' guardada como {csv_path}")

# Uso del script
if __name__ == "__main__":
    excel_path = "Zegoland - parametrized data.xlsx"
    convert_excel_to_csv(excel_path)