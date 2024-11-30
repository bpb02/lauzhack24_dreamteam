import pandas as pd
import os

# Define the Excel files
excel_files = [
    'files/INNOVIX_Elbonie.xlsx',
    'files/BRISTOR_Elbonie.xlsx', 
    'files/INNOVIX_Foresland.xlsx'
]

# Dictionary to store dataframes by sheet name
sheet_data = {}

# Read each Excel file
for file in excel_files:
    if os.path.exists(file):
        xls = pd.ExcelFile(file)
        
        # Process each sheet
        for sheet in xls.sheet_names:
            if sheet.lower() != 'main':  # Skip 'Main' sheet
                df = pd.read_excel(file, sheet_name=sheet)
                
                # Rename 'unit of measure' column if it exists
                df.rename(columns={'unit of measure': 'Measure'}, inplace=True)
                
                # If sheet already exists in dictionary, append the data
                if sheet in sheet_data:
                    sheet_data[sheet].append(df)
                else:
                    sheet_data[sheet] = [df]
    else:
        print(f"Warning: File {file} not found")

# Combine and save data for each sheet
for sheet_name, dfs in sheet_data.items():
    if len(dfs) > 0:
        # Combine all dataframes for this sheet
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save to Excel
        output_file = f'files/combined_{sheet_name}.xlsx'
        combined_df.to_excel(output_file, index=False)
        print(f"Saved combined data for sheet '{sheet_name}' to {output_file}")
