#!/usr/bin/env python3
"""
Disease Outbreak Prediction Training Module

This module handles the training of machine learning models for disease outbreak prediction.
It orchestrates data loading, preprocessing, model training, and saving.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the src directory to Python path to import modules
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
    from disease_outbreak_prediction.preprocessing import preprocess_data
    from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data, train_lstm_model
    from disease_outbreak_prediction.models.spatial_analysis import spatial_feature_importance
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed and modules are available.")
    sys.exit(1)

def main():
    """Main training function"""
    print("🦠 Disease Outbreak Prediction - Training Module")
    print("=" * 50)
    
    try:
        # Create necessary directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        print("📊 Loading data...")
        disease_data = fetch_disease_data()
        print(f"✅ Disease data loaded: {len(disease_data)} records")
        
        climate_data = fetch_climate_data("San Juan", "2010-01-01", "2020-12-31")
        print(f"✅ Climate data loaded: {len(climate_data)} records")
        
        pop_data = get_population_data()
        print(f"✅ Population data loaded: {len(pop_data)} records")
        
        print("\n🔄 Preprocessing data...")
        processed_data = preprocess_data(disease_data, climate_data, pop_data)
        print(f"✅ Data preprocessed: {len(processed_data)} records")
        
        print(f"📈 Data shape: {processed_data.shape}")
        print(f"📅 Date range: {processed_data['date'].min()} to {processed_data['date'].max()}")
        
        print("\n🧠 Training LSTM model...")
        X, y, scaler = prepare_lstm_data(processed_data)
        print(f"✅ LSTM data prepared: X={X.shape}, y={y.shape}")
        
        lstm_model = train_lstm_model(X, y, epochs=50, batch_size=32)
        print("✅ LSTM model trained and saved")
        
        print("\n🌍 Training Random Forest model...")
        rf_model, importance = spatial_feature_importance(processed_data)
        print("✅ Random Forest model trained and saved")
        
        print("\n📊 Feature Importance:")
        print("-" * 30)
        for _, row in importance.iterrows():
            print(f"{row['feature']:20} {row['importance']:.4f}")
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved in: {os.path.abspath('models')}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()