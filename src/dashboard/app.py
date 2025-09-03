import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys

# Add src to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
from disease_outbreak_prediction.preprocessing import preprocess_data
from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data, train_lstm_model, load_lstm_model
from disease_outbreak_prediction.models.spatial_analysis import spatial_feature_importance, load_rf_model

st.title("Disease Outbreak Prediction System")

# Load data
disease_data = fetch_disease_data()
climate_data = fetch_climate_data("San Juan", "2010-01-01", "2020-12-31")
pop_data = get_population_data()

# Preprocess data
processed_data = preprocess_data(disease_data, climate_data, pop_data)

# Train or load models
if st.button("Train Models"):
    # Prepare data for LSTM
    X, y, scaler = prepare_lstm_data(processed_data)
    lstm_model = train_lstm_model(X, y)
    
    # Train Random Forest
    rf_model, importance = spatial_feature_importance(processed_data)
    
    st.success("Models trained successfully!")
else:
    # Try to load existing models
    lstm_model = load_lstm_model()
    rf_model = load_rf_model()
    
    if lstm_model is None or rf_model is None:
        st.warning("Models not found. Please train the models first.")
        st.stop()

# Make predictions
if lstm_model and rf_model:
    # Prepare data for prediction
    X, _, scaler = prepare_lstm_data(processed_data)
    predictions = lstm_model.predict(X)
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(
        np.concatenate([predictions.reshape(-1,1), X[:,-1,1:]], axis=1)
    )[:,0]
    
    # Create results DataFrame
    results = processed_data.copy()
    results = results.iloc[4:]  # Because of look_back
    results['predicted'] = predictions
    
    # Plot predictions vs actual
    fig = px.line(
        results, 
        x='date', 
        y=['cases', 'predicted'],
        title="Outbreak Prediction vs Actual Cases"
    )
    st.plotly_chart(fig)
    
    # Show feature importance
    st.subheader("Feature Importance")
    _, importance = spatial_feature_importance(processed_data)
    st.bar_chart(importance.set_index('feature'))
    
    # Alert system
    if predictions[-1] > 100:
        st.error("⚠️ HIGH RISK: Potential outbreak detected!")