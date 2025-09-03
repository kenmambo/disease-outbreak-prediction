import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

try:
    from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
    from disease_outbreak_prediction.preprocessing import preprocess_data
    from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data, train_lstm_model, load_lstm_model
    from disease_outbreak_prediction.models.spatial_analysis import spatial_feature_importance, load_rf_model
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all dependencies are installed and the project structure is correct.")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Disease Outbreak Prediction System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🦠 Disease Outbreak Prediction System</h1>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("🎛️ Control Panel")

# Progress tracking
progress_placeholder = st.empty()

try:
    # Load data with progress indication
    with st.spinner("📊 Loading data..."):
        disease_data = fetch_disease_data()
        climate_data = fetch_climate_data("San Juan", "2010-01-01", "2020-12-31")
        pop_data = get_population_data()
    
    st.sidebar.success(f"✅ Data loaded: {len(disease_data)} records")
    
    # Preprocess data
    with st.spinner("🔄 Preprocessing data..."):
        processed_data = preprocess_data(disease_data, climate_data, pop_data)
    
    st.sidebar.success(f"✅ Data preprocessed: {len(processed_data)} records")
    
    # Model training/loading section
    st.sidebar.subheader("🤖 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Data Overview")
        st.metric("Total Records", len(processed_data))
        st.metric("Date Range", f"{processed_data['date'].min().strftime('%Y-%m-%d')} to {processed_data['date'].max().strftime('%Y-%m-%d')}")
        
        # Show recent data
        st.subheader("📊 Recent Cases")
        recent_data = processed_data.tail(10)[['date', 'cases', 'temperature', 'humidity']]
        st.dataframe(recent_data, width='stretch')
    
    with col2:
        # Train or load models
        train_button = st.sidebar.button("🚀 Train New Models", type="primary")
        
        if train_button:
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            try:
                # Train LSTM model
                status_text.text("🧠 Training LSTM model...")
                progress_bar.progress(25)
                X, y, scaler = prepare_lstm_data(processed_data)
                lstm_model = train_lstm_model(X, y, epochs=50, batch_size=32)
                progress_bar.progress(50)
                
                # Train Random Forest
                status_text.text("🌍 Training Random Forest model...")
                rf_model, importance = spatial_feature_importance(processed_data)
                progress_bar.progress(100)
                
                status_text.text("✅ Training completed!")
                st.sidebar.success("Models trained successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")
                st.stop()
        else:
            # Try to load existing models
            with st.spinner("📥 Loading existing models..."):
                lstm_model = load_lstm_model()
                rf_model = load_rf_model()
            
            if lstm_model is None or rf_model is None:
                st.warning("⚠️ Models not found. Please train the models first using the sidebar.")
                st.sidebar.warning("📋 No trained models found")
                st.stop()
            else:
                st.sidebar.success("✅ Models loaded successfully")
    
    # Make predictions if models are available
    if 'lstm_model' in locals() and 'rf_model' in locals() and lstm_model is not None and rf_model is not None:
        
        # Prepare data for prediction
        X, _, scaler = prepare_lstm_data(processed_data)
        
        with st.spinner("🔮 Making predictions..."):
            predictions = lstm_model.predict(X, verbose=0)
            
            # Inverse transform predictions
            predictions_reshaped = predictions.reshape(-1, 1)
            dummy_features = X[:, -1, 1:]  # Get the last timestep features
            combined = np.concatenate([predictions_reshaped, dummy_features], axis=1)
            predictions_inverse = scaler.inverse_transform(combined)[:, 0]
        
        # Create results DataFrame
        results = processed_data.copy()
        results = results.iloc[4:].reset_index(drop=True)  # Remove first 4 rows due to look_back
        results['predicted'] = predictions_inverse
        
        # Main dashboard content
        st.subheader("🎯 Outbreak Predictions vs Actual Cases")
        
        # Plot predictions vs actual
        fig = px.line(
            results.tail(365),  # Show last year
            x='date', 
            y=['cases', 'predicted'],
            title="Disease Outbreak Prediction (Last 365 Days)",
            labels={'value': 'Number of Cases', 'date': 'Date'},
            color_discrete_map={'cases': '#1f77b4', 'predicted': '#ff7f0e'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, width='stretch')
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Feature Importance")
            _, importance = spatial_feature_importance(processed_data)
            
            fig_importance = px.bar(
                importance, 
                x='importance', 
                y='feature',
                orientation='h',
                title="Factors Contributing to Disease Spread",
                labels={'importance': 'Importance Score', 'feature': 'Features'}
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, width='stretch')
        
        with col2:
            st.subheader("⚠️ Risk Assessment")
            
            # Current risk assessment
            latest_prediction = predictions_inverse[-1]
            latest_actual = results['cases'].iloc[-1]
            
            # Risk level determination
            if latest_prediction > 150:
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
            elif latest_prediction > 100:
                risk_level = "🟡 MEDIUM RISK"  
                risk_color = "orange"
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
            
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 0.5rem; background-color: {risk_color}20; border-left: 4px solid {risk_color};">
                <h3 style="color: {risk_color}; margin: 0;">{risk_level}</h3>
                <p style="margin: 0.5rem 0;"><strong>Predicted Cases:</strong> {latest_prediction:.0f}</p>
                <p style="margin: 0.5rem 0;"><strong>Actual Cases:</strong> {latest_actual:.0f}</p>
                <p style="margin: 0.5rem 0;"><strong>Prediction Accuracy:</strong> {100 - abs(latest_prediction - latest_actual) / latest_actual * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Alert system
            if latest_prediction > 150:
                st.error("🚨 CRITICAL ALERT: High outbreak risk detected! Immediate action recommended.")
            elif latest_prediction > 100:
                st.warning("⚠️ WARNING: Elevated outbreak risk. Monitor closely.")
            else:
                st.success("✅ NORMAL: Low outbreak risk. Continue regular monitoring.")
        
        # Summary statistics
        st.subheader("📊 Model Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate metrics
        mae = np.mean(np.abs(results['predicted'] - results['cases']))
        rmse = np.sqrt(np.mean((results['predicted'] - results['cases'])**2))
        accuracy = 100 - (mae / np.mean(results['cases']) * 100)
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f}")
        with col3:
            st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
        with col4:
            st.metric("Total Predictions", len(results))

except Exception as e:
    st.error(f"❌ Application error: {e}")
    import traceback
    st.code(traceback.format_exc())
    
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🦠 Disease Outbreak Prediction System | Built with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)