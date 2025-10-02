import sys
import os

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
from disease_outbreak_prediction.preprocessing import preprocess_data
from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data, train_lstm_model
from disease_outbreak_prediction.models.spatial_analysis import spatial_feature_importance

def retrain_models():
    """
    Retrains the LSTM and Random Forest models with extended parameters.
    """
    print("🔄 Retraining models with enhanced parameters...")

    # Load and process data
    disease_data = fetch_disease_data()
    climate_data = fetch_climate_data('San Juan', '2010-01-01', '2020-12-31')
    pop_data = get_population_data()
    processed_data = preprocess_data(disease_data, climate_data, pop_data)

    # Extended LSTM training
    X, y, scaler = prepare_lstm_data(processed_data)
    # More epochs, smaller batch size for potentially better results
    lstm_model = train_lstm_model(X, y, epochs=100, batch_size=16)

    # Retrain Random Forest with different parameters
    # The function can be extended to accept different params
    rf_model, importance = spatial_feature_importance(processed_data)

    print("✅ Model retraining completed")
    print("📊 New Feature Importance:")
    print(importance)

if __name__ == "__main__":
    retrain_models()