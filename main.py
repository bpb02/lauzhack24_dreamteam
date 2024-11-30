import streamlit as st
import pandas as pd
import plotly.express as px

# Read the Excel file
excel_file = 'files/INNOVIX_Elbonie.xlsx'
xls = pd.ExcelFile(excel_file)

# Get all sheet names except 'Main'
sheet_names = [sheet for sheet in xls.sheet_names if sheet.lower() != 'main']

# Create a sheet selector
selected_sheet = st.selectbox('Select a sheet:', sheet_names)

# Read the selected sheet
df = pd.read_excel(excel_file, sheet_name=selected_sheet)

# Display basic information about the data
st.write("### Data Overview")
st.write(f"Number of rows: {len(df)}")
st.write(f"Number of columns: {len(df.columns)}")
st.write("\n### Column Information")
st.write(df.dtypes)

# Display summary statistics
st.write("\n### Summary Statistics")
st.write(df.describe())

# Display distinct values for object columns
st.write("\n### Distinct Values in Categorical Columns")
object_columns = df.select_dtypes(include=['object']).columns
for col in object_columns:
    unique_values = df[col].unique()[:20]  # Get up to 20 distinct values
    st.write(f"\n**{col}** - {len(unique_values)} distinct values shown:")
    st.write(unique_values)

# Create distribution plots for numeric columns
st.write("\n### Distribution Plots")
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    fig = px.histogram(df, x=col, title=f'Distribution of {col}')
    st.plotly_chart(fig)
    
    # Add box plot for better distribution visualization
    fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
    st.plotly_chart(fig_box)

# Display sample of the data
st.write("\n### Sample Data")
st.write(df.head())
