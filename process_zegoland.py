import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    # Create plots directory if it doesn't exist
    os.makedirs('plots/zegoland', exist_ok=True)

    # Read and process the data
    df = pd.read_csv('data/Zegoland_parametrized_data.csv', sep=';', decimal=',')
    # Create a mapping of Spanish month abbreviations to numbers
    month_map = {
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'ago': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }
    
    # Convert date strings to datetime
    df['date'] = df['date'].str.strip().apply(lambda x: pd.to_datetime('20' + x.split('-')[1] + '-' + 
                                                                      month_map[x.split('-')[0]] + '-01'))
    print("ok")
    df = df.sort_values('date')
    
    # Clean numeric columns by removing spaces and converting to float
    numeric_cols = df.columns.drop('date')
    for col in numeric_cols:
        # Check if column contains string values before using str accessor
        if df[col].dtype == 'object':
            # Replace comma with period for decimal numbers
            df[col] = df[col].str.strip().str.replace(' ', '').str.replace('.', '').str.replace('%', '').str.replace(',', '.').astype(float)
        else:
            # If already numeric, just ensure it's float type
            df[col] = df[col].astype(float)

    print("ok2")

    # Create time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Select features for modeling
    features = ['year', 'month', 'ind A demand volume Bristor', 'ind B demand volume Bristor',
                'New paitient #, ind A, Bristor', 'New paitient #, ind B, Bristor',
                'email #', 'face to face call #', 'mail #', 'remote call #', 'telephone #',
                'bristor share of voice % ind A', 'ind A demand volume competitor',
                'competitor share voice % ind A', 'bristor share voice % ind B',
                'competitor share voice % ind B']
    
    X = df[features]
    y = df['truth_sales']
    
    # Plot correlation matrix
    plt.figure(figsize=(15, 12))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/zegoland/correlation_matrix.png')
    plt.close()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Time series cross validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Train XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # Calculate cross-validation scores
    mse_scores = []
    mape_scores = []
    
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        mse_scores.append(mse)
        mape_scores.append(mape)
    
    print(f"\nAverage MSE: {np.mean(mse_scores):,.2f}")
    print(f"Average MAPE: {np.mean(mape_scores):.2f}%")
    
    # Train final model on all data
    xgb_model.fit(X_scaled, y)
    
    # Generate next two months dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=30), periods=2, freq='ME')
    
    # Create future feature set
    future_data = []
    for date in future_dates:
        if len(future_data) > 0:
            prev_prediction = xgb_model.predict(scaler.transform(pd.DataFrame([future_data[-1]])[features]))[0]
        else:
            prev_prediction = df['truth_sales'].iloc[-1]
            
        prev_row = df.iloc[-1] if len(future_data) == 0 else pd.Series(future_data[-1])
        
        future_row = {
            'year': date.year,
            'month': date.month
        }
        
        # Copy other features with small random variations
        for feat in features[2:]:  # Skip year and month
            future_row[feat] = prev_row[feat] * (1 + np.random.normal(0, 0.1))
            
        future_data.append(future_row)
    
    future_df = pd.DataFrame(future_data)
    future_scaled = scaler.transform(future_df[features])
    future_predictions = xgb_model.predict(future_scaled)
    
    # Create results DataFrame
    future_results = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Sales': future_predictions
    })
    
    print("\nPredictions for next two months:")
    print(future_results.to_string(index=False))
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], y, label='Actual Sales', color='blue')
    plt.scatter(future_dates, future_predictions, color='red', label='Predictions')
    plt.title('Sales Predictions')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/zegoland/predictions_plot.png')
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': xgb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 5 most important features:")
    print(feature_importance.head().to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('plots/zegoland/feature_importance.png')
    plt.close()

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check that the data file exists and contains the expected columns")
