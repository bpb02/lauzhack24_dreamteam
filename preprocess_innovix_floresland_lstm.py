import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

    X = df[features].values
    y = df['Ex_Factory_Volume'].values

    # Scale the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Create sequences for LSTM
    def create_sequences(X, y, seq_length=3):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)

    seq_length = 3
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

    # Split data into train and test
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]

    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, X.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Generate future predictions
    last_sequence = X_scaled[-seq_length:]
    future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=30), periods=2, freq='ME')
    future_predictions = []

    for date in future_dates:
        # Predict next value
        next_pred = model.predict(last_sequence.reshape(1, seq_length, X.shape[1]))
        
        # Create next feature set
        next_features = {
            'Year': date.year,
            'Month': date.month,
            'YREX_demand_mg': scaler_y.inverse_transform(next_pred)[0][0] * 0.4,
            'YREX_demand_mot': scaler_y.inverse_transform(next_pred)[0][0] * 0.3,
            'Email_activity': X[-1, features.index('Email_activity')] * (1 + np.random.normal(0, 0.1)),
            'Remote_call_activity': X[-1, features.index('Remote_call_activity')] * (1 + np.random.normal(0, 0.1)),
            'F2F_call_activity': X[-1, features.index('F2F_call_activity')] * (1 + np.random.normal(0, 0.1)),
            'Meetings_activity': X[-1, features.index('Meetings_activity')] * (1 + np.random.normal(0, 0.1)),
            'INNOVIX_Patient_Share_Mean': X[-1, features.index('INNOVIX_Patient_Share_Mean')],
            'YREX_Patient_Share_Mean': X[-1, features.index('YREX_Patient_Share_Mean')]
        }
        
        # Scale new features
        next_features_scaled = scaler_X.transform(pd.DataFrame([next_features])[features])
        
        # Update sequence for next prediction
        last_sequence = np.vstack((last_sequence[1:], next_features_scaled))
        
        # Store prediction
        future_predictions.append(scaler_y.inverse_transform(next_pred)[0][0])

    # Create results DataFrame
    future_results = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Ex_Factory_Volume': future_predictions
    })

    # Calculate metrics
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100

    print(f"\nTest RMSE: {rmse:,.2f}")
    print(f"Test MAPE: {mape:.2f}%")

    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Historical Data and Predictions
    plt.subplot(2, 2, 1)
    plt.plot(df['Date'], df['Ex_Factory_Volume'], label='Historical', color='blue')
    plt.plot(future_results['Date'], future_results['Predicted_Ex_Factory_Volume'], 
             label='Predictions', color='red', linestyle='--')
    plt.title('Ex Factory Volume Forecast (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot 2: Training Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
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
    plt.savefig('analysis_plots_innovix_floresland_lstm.png')
    print("\nAnalysis plots saved as 'analysis_plots_innovix_floresland_lstm.png'")

    # Print predictions
    print("\nPredictions for November and December 2024:")
    print(future_results.to_string(index=False))

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check that:")
    print("1. The data file exists at 'data/floresland/processed_floresland_data.csv'")
    print("2. The data file contains all required columns")
    print("3. The data types are correct")
