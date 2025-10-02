import sys
import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add src to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data
from disease_outbreak_prediction.preprocessing import preprocess_data
from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data, train_lstm_model
from disease_outbreak_prediction.models.spatial_analysis import spatial_feature_importance

def evaluate_models():
    """
    Train and evaluate LSTM and Random Forest models, save performance,
    and set outputs for GitHub Actions.
    """
    print("📊 Loading and preprocessing data...")
    disease_data = fetch_disease_data()
    climate_data = fetch_climate_data('San Juan', '2010-01-01', '2020-12-31')
    pop_data = get_population_data()
    processed_data = preprocess_data(disease_data, climate_data, pop_data)

    print("🧠 Training LSTM model...")
    X, y, scaler = prepare_lstm_data(processed_data)

    # Split data for evaluation
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train LSTM
    lstm_model = train_lstm_model(X_train, y_train, epochs=20, batch_size=32)

    # Evaluate LSTM
    lstm_predictions = lstm_model.predict(X_test, verbose=0)
    lstm_mae = mean_absolute_error(y_test, lstm_predictions)
    lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
    lstm_accuracy = max(0, 100 - (lstm_mae / np.mean(y_test) * 100))

    print("🌍 Training Random Forest model...")
    rf_model, importance = spatial_feature_importance(processed_data)

    # Evaluate Random Forest on test set
    test_data = processed_data.iloc[split_idx:].copy()
    available_features = [f for f in ['temperature', 'humidity', 'population_density', 'cases_lag_4w'] if f in test_data.columns]

    if available_features:
        X_rf_test = test_data[available_features].fillna(test_data[available_features].mean())
        y_rf_test = test_data['cases'].fillna(test_data['cases'].mean())

        rf_predictions = rf_model.predict(X_rf_test)
        rf_mae = mean_absolute_error(y_rf_test, rf_predictions)
        rf_rmse = np.sqrt(mean_squared_error(y_rf_test, rf_predictions))
        rf_accuracy = max(0, 100 - (rf_mae / np.mean(y_rf_test) * 100))
    else:
        rf_accuracy, rf_mae, rf_rmse = 0, 0, 0

    # Determine if retraining is needed
    needs_retraining = lstm_accuracy < 85.0 or rf_accuracy < 80.0

    # Save results
    results = {
        'lstm_accuracy': round(lstm_accuracy, 2), 'lstm_mae': round(lstm_mae, 4), 'lstm_rmse': round(lstm_rmse, 4),
        'rf_accuracy': round(rf_accuracy, 2), 'rf_mae': round(rf_mae, 4), 'rf_rmse': round(rf_rmse, 4),
        'needs_retraining': needs_retraining,
        'feature_importance': importance.to_dict('records') if not importance.empty else []
    }

    with open('model_performance.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"📊 LSTM Accuracy: {lstm_accuracy:.2f}%")
    print(f"📊 RF Accuracy: {rf_accuracy:.2f}%")
    print(f"🔄 Needs Retraining: {needs_retraining}")

    # Set outputs for GitHub Actions
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write(f"lstm-accuracy={lstm_accuracy:.2f}\n")
            f.write(f"rf-accuracy={rf_accuracy:.2f}\n")
            f.write(f"needs-retraining={str(needs_retraining).lower()}\n")

if __name__ == "__main__":
    evaluate_models()