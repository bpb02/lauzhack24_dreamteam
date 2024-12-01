import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV files
activity_df = pd.read_csv('files/combined_Activity.csv')
demand_df = pd.read_csv('files/combined_Demand volumes.csv')

# Convert Date columns to datetime
activity_df['Date'] = pd.to_datetime(activity_df['Date'])
demand_df['Date'] = pd.to_datetime(demand_df['Date'])

# Pivot activity data to get channels as columns
activity_pivot = activity_df.pivot_table(
    index=['Date', 'Product', 'Country'],
    columns='Channel',
    values='Value',
    aggfunc='sum'
).reset_index()

# Merge activity and demand data
merged_df = pd.merge(
    activity_pivot,
    demand_df[['Date', 'Product', 'Country', 'Value']],
    on=['Date', 'Product', 'Country'],
    how='inner'
)

# Rename the demand Value column
merged_df = merged_df.rename(columns={'Value': 'Demand'})

# Calculate correlations
correlation_matrix = merged_df.corr()

# Create correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Activity Channels and Demand')
plt.tight_layout()
plt.show()

# Print correlation values with Demand specifically
print("\nCorrelations with Demand:")
for column in correlation_matrix.index:
    if column != 'Demand':
        print(f"{column}: {correlation_matrix.loc[column, 'Demand']:.3f}")
