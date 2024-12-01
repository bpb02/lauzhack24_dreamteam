import pandas as pd
import os
from datetime import datetime

def standardize_date(date_str):
    """Convert various date formats to standard datetime"""
    date_str = str(date_str).lower().strip()
    
    # Map French month abbreviations to English
    month_map = {
        'janv': 'jan', 'févr': 'feb', 'mars': 'mar', 'avr': 'apr',
        'mai': 'may', 'juin': 'jun', 'juil': 'jul', 'août': 'aug',
        'sept': 'sep', 'oct': 'oct', 'nov': 'nov', 'déc': 'dec'
    }
    
    for fr, en in month_map.items():
        date_str = date_str.replace(fr, en)
    
    # Remove dots and clean up
    date_str = date_str.replace('.', '').replace('-', ' ')
    
    try:
        return pd.to_datetime(date_str, format='%b %y')
    except:
        return pd.to_datetime(date_str)

def read_excel_file(file_path):
    """Read data from Excel file"""
    sheet_data = {}
    
    if os.path.exists(file_path):
        xls = pd.ExcelFile(file_path)
        
        for sheet in ['Ex-Factory volumes', 'Activity', 'Demand volumes', 'New patient share']:
            if sheet in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet)
                df['Source_File'] = os.path.basename(file_path)
                df['Country'] = df['Source_File'].str.split('_').str[1].str.split('.').str[0]
                sheet_data[sheet] = df
    else:
        print(f"Warning: File {file_path} not found")
            
    return sheet_data

def transform_dataframe(df):
    # Drop rows where 'Product' is empty or NaN
    df = df.dropna(subset=['Product'])
    
    # Get unique dates
    base_columns = ['Date', 'Main_product', 'Ex_Factory_Volume', 'demand_competence']
    
    # Pivot the DataFrame to create product-specific columns
    df_final = df.pivot_table(
        index=base_columns,
        columns='Product',
        values=['Email', 'Face to face call', 'Meeting'],
        aggfunc='first'
    ).reset_index()
    
    # Flatten and rename columns
    df_final.columns = [
        f"{col[1]}_{col[0]}" if col[1] else col[0] 
        for col in df_final.columns
    ]
    
    # Clean up column names
    df_final.columns = df_final.columns.str.replace('nan_', '')
    
    return df_final

def process_and_merge_data(sheet_data):
    """Process and merge Ex-Factory volumes with Activity and Demand volumes data"""
    # Get individual dataframes
    ex_factory_df = sheet_data['Ex-Factory volumes']
    activity_df = sheet_data['Activity']
    demand_df = sheet_data['Demand volumes']
    patient_share_df = sheet_data['New patient share']
    
    # Standardize dates
    ex_factory_df['Date'] = ex_factory_df['Date'].apply(standardize_date)
    activity_df['Date'] = activity_df['Date'].apply(standardize_date)
    demand_df['Date'] = demand_df['Date'].apply(standardize_date)
    patient_share_df['Date'] = patient_share_df['Date'].apply(standardize_date)
    # Convert percentage string to float for patient share
    patient_share_df['Value'] = patient_share_df['Value'].astype(str).str.rstrip('%') \
                                                        .str.replace(',', '.') \
                                                        .astype(float)

    # Filter non-zero values and calculate mean by date for patient share - separate by product
    innovix_share = patient_share_df[
        (patient_share_df['Value'] != 0) & 
        (patient_share_df['Product'] == 'INNOVIX')
    ].groupby('Date')['Value'].mean().reset_index().rename(columns={'Value': 'INNOVIX_Patient_Share_Mean'})

    yrex_share = patient_share_df[
        (patient_share_df['Value'] != 0) & 
        (patient_share_df['Product'] == 'YREX')
    ].groupby('Date')['Value'].mean().reset_index().rename(columns={'Value': 'YREX_Patient_Share_Mean'})

    # Rename ex-factory Value column
    ex_factory_df = ex_factory_df.rename(columns={'Value': 'Ex_Factory_Volume'})
    ex_factory_df = ex_factory_df.rename(columns={'Product': 'Main_product'})

    # Create filtered dataframes for each demand case
    innovix_mg = demand_df[
        (demand_df['Product'] == 'INNOVIX') & 
        (demand_df['Unit of measure'] == 'Milligrams')
    ][['Date', 'Value']].rename(columns={'Value': 'INNOVIX_demand_mg'})

    innovix_mot = demand_df[
        (demand_df['Product'] == 'INNOVIX') & 
        (demand_df['Unit of measure'] == 'Month of treatment')
    ][['Date', 'Value']].rename(columns={'Value': 'INNOVIX_demand_mot'})

    yrex_mg = demand_df[
        (demand_df['Product'] == 'YREX') & 
        (demand_df['Unit of measure'] == 'Milligrams')
    ][['Date', 'Value']].rename(columns={'Value': 'YREX_demand_mg'})

    yrex_mot = demand_df[
        (demand_df['Product'] == 'YREX') & 
        (demand_df['Unit of measure'] == 'Month of treatment')
    ][['Date', 'Value']].rename(columns={'Value': 'YREX_demand_mot'})

    # Create filtered activity dataframes for each channel
    email_activity = activity_df[
        activity_df['Channel'] == 'Email'
    ].groupby('Date')['Value'].sum().reset_index().rename(columns={'Value': 'Email_activity'})

    remote_activity = activity_df[
        activity_df['Channel'] == 'Remote call'
    ].groupby('Date')['Value'].sum().reset_index().rename(columns={'Value': 'Remote_call_activity'})

    f2f_activity = activity_df[
        activity_df['Channel'] == 'Face to face call'
    ].groupby('Date')['Value'].sum().reset_index().rename(columns={'Value': 'F2F_call_activity'})

    meetings_activity = activity_df[
        activity_df['Channel'] == 'Meetings'
    ].groupby('Date')['Value'].sum().reset_index().rename(columns={'Value': 'Meetings_activity'})

    # Merge all demand dataframes with ex_factory_df
    merged_df = ex_factory_df[['Date', 'Main_product', 'Ex_Factory_Volume']]
    
    # Merge demand dataframes
    for df in [innovix_mg, innovix_mot, yrex_mg, yrex_mot]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='Date',
            how='left'
        )
    
    # Merge activity dataframes and patient share means
    for df in [email_activity, remote_activity, f2f_activity, meetings_activity, innovix_share, yrex_share]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='Date',
            how='left'
        )
    
    # Remove rows where more than half of columns have missing values
    threshold = len(merged_df.columns) / 2
    merged_df = merged_df.dropna(thresh=threshold)
    
    # Fill remaining NaN values with 0
    merged_df = merged_df.fillna(0)

    

    # Rename ex-factory Value column
    # merged_df = merged_df.rename(columns={'INNOVIX': 'demand_INNOVIX'})
    # merged_df = merged_df.rename(columns={'YREX': 'demand_competence'})

    # Drop demand_innovix column since it duplicates ex_factory_volume
    # merged_df = merged_df.drop(columns=['demand_INNOVIX'])

    # Pivot activity data to create separate columns for each product-channel combination
    # activity_pivot = activity_df.pivot_table(
    #     index=['Date', 'Product', 'Country'],
    #     columns='Channel',
    #     values='Value',
    #     aggfunc='sum'
    # ).reset_index()
    
    # Second merge: Add activity data
    # final_df = pd.merge(
    #     merged_df,
    #     activity_pivot,
    #     on=['Date'],
    #     how='inner'
    # )

    # final_df = transform_dataframe(final_df)
    
    # Rename demand Value column
    # final_df = final_df.rename(columns={'Value': 'Demand_Volume'})
    
    return merged_df

def main():
    # Define Excel file
    excel_file = 'data/INNOVIX_Floresland.xlsx'
    
    # Create output directory if it doesn't exist
    os.makedirs('data/floresland', exist_ok=True)
    
    # Read data
    sheet_data = read_excel_file(excel_file)
    
    # Process and merge data
    final_df = process_and_merge_data(sheet_data)
    
    # Save to Excel
    output_excel = 'data/floresland/processed_floresland_data.xlsx'
    output_csv = 'data/floresland/processed_floresland_data.csv'
    final_df.to_excel(output_excel, index=False)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved processed data to {output_excel} and {output_csv}")

if __name__ == "__main__":
    main()
