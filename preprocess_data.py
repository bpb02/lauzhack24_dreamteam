import pandas as pd
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

# Read each Excel file
for file in excel_files:
    if os.path.exists(file):
        xls = pd.ExcelFile(file)
        
        # Process Activity and Demand volumes sheets
        for sheet in ['Activity', 'Demand volumes']:
            if sheet in xls.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet)
                
                # Rename 'unit of measure' column if it exists
                df.rename(columns={'unit of measure': 'Measure', 'Unit of measure': 'Measure'}, inplace=True)
                
                # Add file source column
                df['Source_File'] = os.path.basename(file)
                
                # Append to sheet data
                sheet_data[sheet].append(df)
    else:
        print(f"Warning: File {file} not found")

# Combine and save data for each sheet
for sheet_name, dfs in sheet_data.items():
    if len(dfs) > 0:
        # Combine all dataframes for this sheet
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save to CSV
        output_file = f'files/combined_{sheet_name}.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"Saved combined data for sheet '{sheet_name}' to {output_file}")
