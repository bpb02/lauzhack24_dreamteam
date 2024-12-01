import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import plotly.express as px
import plotly.graph_objects as go

class ZegolandSalesForecast:
    def __init__(self, file_path):
        """
        Initialize the sales forecast class with data preprocessing and model preparation.
        
        Args:
            file_path (str): Path to the Excel file containing sales and feature data
        """
        self.file_path = file_path
        self.df_processed = None
        self.feature_weights = None
        self.prophet_model = None
        self.forecast = None
        
        # Normalization with standardization and winsorization
        self.scaler = StandardScaler()
        self.winsorizer = FunctionTransformer(lambda x: np.clip(x, np.percentile(x, 5), np.percentile(x, 95)), validate=False)

    def load_and_preprocess_data(self):
        """
        Load and preprocess data, creating normalized features with reduced sensitivity to outliers.
        
        Returns:
            pd.DataFrame: Processed dataframe with engineered features
            dict: Weights for different features
        """
        # Read data
        df = pd.read_excel(self.file_path)
        df.columns = ['ds', 'y'] + list(df.columns[2:])
        df['ds'] = pd.to_datetime(df['ds'], format='%b-%y')
        
        # Feature weighting based on potential impact
        feature_weights = {
            'ind A demand volume': 0.15,
            'Bristorind B demand volume': 0.15, 
            'Bristor share of voice % ind A': 0.2,
            'competitor share voice % ind A': 0.1,
            'competitor share voice % ind B': 0.1,
            'total paitient': 0.1,
            'New paitient': 0.1,
            'email #': 0.05,
            'face to face': 0.05,
            'remote call #': 0.05,
            'telephone #': 0.05
        }
        
        # Standardize and winsorize features
        for feature in feature_weights.keys():
            if feature in df.columns:
                # Standardize and winsorize the feature
                df[f'{feature}_normalized'] = self.winsorizer.transform(self.scaler.fit_transform(df[[feature]]))
                
                # Create lagged and rolling features with reduced outlier sensitivity
                df[f'{feature}_lag1'] = df[f'{feature}_normalized'].shift(1)
                df[f'{feature}_rolling_mean'] = df[f'{feature}_normalized'].rolling(window=3, min_periods=1).mean()
        
        self.df_processed = df.dropna()
        self.feature_weights = feature_weights
        
        return self.df_processed, feature_weights

    def create_prophet_features(self):
        """
        Prepare dataframe with additional regressors for Prophet model.
        
        Returns:
            pd.DataFrame: Dataframe with Prophet-ready features
        """
        prophet_df = self.df_processed[['ds', 'y']].copy()
        
        for feature in self.feature_weights.keys():
            normalized_cols = [
                col for col in self.df_processed.columns 
                if col.startswith(f'{feature}_normalized') or 
                   col.startswith(f'{feature}_lag') or 
                   col.startswith(f'{feature}_rolling')
            ]
            
            for col in normalized_cols:
                prophet_df[col] = self.df_processed[col]
        
        return prophet_df

    def train_prophet_model(self, smoothing_factor=0.3, changepoint_prior_scale=0.05, interval_width=0.95):
        """
        Train Prophet model with additional regressors and apply smoothing.
        
        Args:
            smoothing_factor (float): Factor to reduce forecast volatility
            changepoint_prior_scale (float): Sensitivity to trend changes
            interval_width (float): Width of confidence intervals
        
        Returns:
            Prophet model and forecast dataframe
        """
        prophet_df = self.create_prophet_features()
        
        model = Prophet(
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=interval_width,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        # Add regressors dynamically
        additional_regressors = [
            col for col in prophet_df.columns 
            if col not in ['ds', 'y'] and 'normalized' in col
        ]
        
        for regressor in additional_regressors:
            model.add_regressor(regressor)
        
        model.fit(prophet_df)
        
        # Create future dataframe with smoothing
        future = model.make_future_dataframe(periods=12, freq='M')
        
        # Add future regressors using last known values with smoothing
        for regressor in additional_regressors:
            future[regressor] = prophet_df[regressor].iloc[-1] * (1 - smoothing_factor)
        
        forecast = model.predict(future)
        
        # Post-process the forecast
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=forecast['yhat'].max() * 1.1)
        
        self.prophet_model = model
        self.forecast = forecast
        
        return model, forecast

def create_streamlit_app(forecast_instance):
    """
    Create a Streamlit web application for sales forecast visualization.
    
    Args:
        forecast_instance (ZegolandSalesForecast): Initialized forecast object
    """
    st.set_page_config(layout="wide")
    st.title("Zegoland Sales Forecast Dashboard")
    
    # Sidebar for user interactions
    st.sidebar.header("Forecast Configuration")
    smoothing = st.sidebar.slider("Forecast Smoothing", 0.0, 1.0, 0.3, 0.1)
    changepoint_scale = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05, 0.01)
    interval_width = st.sidebar.slider("Forecast Interval Width", 0.80, 0.99, 0.95, 0.01)
    
    # Perform forecast with selected parameters
    model, forecast = forecast_instance.train_prophet_model(
        smoothing_factor=smoothing,
        changepoint_prior_scale=changepoint_scale,
        interval_width=interval_width
    )
    
    # Create multiple visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Sales Forecast", "Forecast Components", "Feature Importance", "Forecast Evaluation"])
    
    with tab1:
        st.subheader("Sales Forecast")
        fig = plot_plotly(model, forecast)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Forecast Components")
        fig_components = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_components, use_container_width=True)
    
    with tab3:
        st.subheader("Feature Weights")
        weights_df = pd.DataFrame.from_dict(forecast_instance.feature_weights, orient='index', columns=['Weight'])
        weights_df.index.name = 'Feature'
        
        fig_weights = px.bar(
            weights_df.reset_index(), 
            x='Feature', 
            y='Weight', 
            title='Feature Importance in Sales Forecast'
        )
        st.plotly_chart(fig_weights, use_container_width=True)
    
    with tab4:
        st.subheader("Forecast Evaluation")
        
        # Calculate evaluation metrics
        y_true = forecast_instance.df_processed['y']
        y_pred = forecast['yhat']
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Display evaluation metrics
        st.write(f"MAPE: {mape:.2f}%")
        st.write(f"RMSE: {rmse:.2f}")
        
        # Plot actual vs. predicted
        fig_actual_vs_pred = go.Figure()
        fig_actual_vs_pred.add_trace(go.Scatter(x=forecast_instance.df_processed['ds'], y=y_true, mode='lines', name='Actual'))
        fig_actual_vs_pred.add_trace(go.Scatter(x=forecast['ds'], y=y_pred, mode='lines', name='Predicted'))
        fig_actual_vs_pred.update_layout(title='Actual vs. Predicted Sales')
        st.plotly_chart(fig_actual_vs_pred, use_container_width=True)

def main():
    file_path = 'files/Zegoland - parametrized data (2).xlsx'
    forecast_instance = ZegolandSalesForecast(file_path)
    
    # Preprocess data
    forecast_instance.load_and_preprocess_data()
    
    # Create Streamlit app
    create_streamlit_app(forecast_instance)

if __name__ == '__main__':
    main()