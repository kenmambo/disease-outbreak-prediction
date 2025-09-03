import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
from preprocessing import preprocess_data
from models.lstm_model import prepare_lstm_data, train_lstm_model
from models.spatial_analysis import spatial_feature_importance

def main():
    print("Loading data...")
    disease_data = fetch_disease_data()
    climate_data = fetch_climate_data("San Juan", "2010-01-01", "2020-12-31")
    pop_data = get_population_data()
    
    print("Preprocessing data...")
    processed_data = preprocess_data(disease_data, climate_data, pop_data)
    
    print("Training LSTM model...")
    X, y, _ = prepare_lstm_data(processed_data)
    lstm_model = train_lstm_model(X, y)
    
    print("Training Random Forest model...")
    rf_model, importance = spatial_feature_importance(processed_data)
    
    print("Training completed successfully!")
    print(f"Feature importance:\n{importance}")

if __name__ == "__main__":
    main()