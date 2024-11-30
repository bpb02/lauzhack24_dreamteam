import streamlit as st
import pandas as pd
import plotly.express as px
import os
# Define the Excel files
excel_files = [
    'files/INNOVIX_Elbonie.xlsx',
    'files/INNOVIX_Floresland.xlsx',
    'files/BRISTOR_Zegoland.xlsx'
]

# Dictionary to store dataframes by sheet name
sheet_data = {
    'Activity': [],
    'Demand volumes': []
}

# Read each Excel file and store data by sheet
for file in excel_files:
    try:
        xls = pd.ExcelFile(file)
        for sheet_name in ['Activity', 'Demand volumes']:
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name)
                sheet_data[sheet_name].append({
                    'file': file,
                    'columns': set(df.columns)
                })
    except Exception as e:
        st.error(f"Error reading {file}: {str(e)}")

# Display column comparison for each sheet
st.write("### Column Comparison Analysis")

for sheet_name in ['Activity', 'Demand volumes']:
    st.write(f"\n## {sheet_name} Sheet Analysis")
    
    if not sheet_data[sheet_name]:
        st.warning(f"No {sheet_name} sheets found in any files")
        continue
        
    # Get all unique columns across files
    all_columns = set()
    for data in sheet_data[sheet_name]:
        all_columns.update(data['columns'])
    
    # Create comparison table
    comparison_data = []
    for data in sheet_data[sheet_name]:
        file_name = os.path.basename(data['file'])
        row = {'File': file_name}
        for col in all_columns:
            row[col] = '✓' if col in data['columns'] else '✗'
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.write("Column presence in each file (✓ = present, ✗ = missing):")
    st.dataframe(comparison_df)
    
    # Show summary statistics
    common_columns = set.intersection(*[data['columns'] for data in sheet_data[sheet_name]])
    unique_columns = set.union(*[data['columns'] for data in sheet_data[sheet_name]])
    
    st.write(f"\nSummary for {sheet_name}:")
    st.write(f"- Total unique columns across all files: {len(unique_columns)}")
    st.write(f"- Common columns present in all files: {len(common_columns)}")
    
    if common_columns:
        st.write("\nCommon columns:")
        st.write(sorted(common_columns))
    
    if len(unique_columns) > len(common_columns):
        st.write("\nColumns that differ between files:")
        st.write(sorted(unique_columns - common_columns))
