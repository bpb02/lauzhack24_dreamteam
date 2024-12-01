import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Read the data
    df = pd.read_csv('data/floresland/processed_floresland_data.csv')

    # Convert Date to datetime and sort by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Create time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Select features for modeling
    features = ['Year', 'Month',
                'YREX_demand_mg', 'YREX_demand_mot',
                'Email_activity', 'Remote_call_activity', 
                'F2F_call_activity', 'Meetings_activity',
                'INNOVIX_Patient_Share_Mean', 'YREX_Patient_Share_Mean']

    X = df[features]
    y = df['Ex_Factory_Volume']

    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest model using all data
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)

    # Generate dates for November and December 2024
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=30), 
                               periods=2, 
                               freq='ME')
    
    # Create future feature set using model predictions
    future_data = []
    for date in future_dates:
        # Get previous prediction to use as input for next prediction
        if len(future_data) > 0:
            prev_prediction = rf_model.predict(scaler.transform(pd.DataFrame([future_data[-1]])[features]))[0]
        else:
            prev_prediction = df['Ex_Factory_Volume'].iloc[-1]
            
        # Use model to predict features for next month
        prev_row = df.iloc[-1] if len(future_data) == 0 else pd.Series(future_data[-1])
        
        future_row = {
            'Year': date.year,
            'Month': date.month,
            'YREX_demand_mg': prev_prediction * 0.4,
            'YREX_demand_mot': prev_prediction * 0.3,
            'Email_activity': prev_row['Email_activity'] * (1 + np.random.normal(0, 0.1)),
            'Remote_call_activity': prev_row['Remote_call_activity'] * (1 + np.random.normal(0, 0.1)),
            'F2F_call_activity': prev_row['F2F_call_activity'] * (1 + np.random.normal(0, 0.1)),
            'Meetings_activity': prev_row['Meetings_activity'] * (1 + np.random.normal(0, 0.1)),
            'INNOVIX_Patient_Share_Mean': prev_row['INNOVIX_Patient_Share_Mean'],
            'YREX_Patient_Share_Mean': prev_row['YREX_Patient_Share_Mean']
        }
        future_data.append(future_row)

    print("ok")

    future_df = pd.DataFrame(future_data)

    # Scale future features
    future_scaled = scaler.transform(future_df[features])

    # Make predictions
    future_predictions = rf_model.predict(future_scaled)

    # Create results DataFrame
    future_results = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Ex_Factory_Volume': future_predictions
    })
    # Calculate cross-validation scores
    rmse_scores = []
    mape_scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        # Calculate RMSE
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_scores.append(rmse)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        mape_scores.append(mape)

    print("\nCross-validation RMSE scores:", rmse_scores)
    print(f"Average RMSE: {np.mean(rmse_scores):,.2f}")
    print("\nCross-validation MAPE scores:", mape_scores)
    print(f"Average MAPE: {np.mean(mape_scores):.2f}%")

    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Historical Data and Predictions
    plt.subplot(2, 2, 1)
    plt.plot(df['Date'], df['Ex_Factory_Volume'], label='Historical', color='blue')
    plt.plot(future_results['Date'], future_results['Predicted_Ex_Factory_Volume'], 
             label='Predictions', color='red', linestyle='--')
    plt.title('Ex Factory Volume Forecast')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Feature Importance
    plt.subplot(2, 2, 2)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.title('Feature Importance')
    
    # Plot 3: Correlation Heatmap
    plt.subplot(2, 2, 3)
    correlation_matrix = df[features + ['Ex_Factory_Volume']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    
    # Plot 4: Monthly Average Volume
    plt.subplot(2, 2, 4)
    monthly_avg = df.groupby('Month')['Ex_Factory_Volume'].mean()
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
    plt.title('Monthly Average Volume')
    plt.xlabel('Month')
    plt.ylabel('Average Volume')
    
    plt.tight_layout()
    plt.savefig('analysis_plots_innovix_floresland.png')
    print("\nAnalysis plots saved as 'analysis_plots_innovix_floresland.png'")

    # Print predictions and feature importance
    print("\nPredictions for November and December 2024:")
    print(future_results.to_string(index=False))
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check that:")
    print("1. The data file exists at 'data/floresland/processed_floresland_data.csv'")
    print("2. The data file contains all required columns")
    print("3. The data types are correct")
