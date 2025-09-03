import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import os

def build_lstm_model(input_shape):
    """Build and compile LSTM model"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_lstm_data(data, look_back=4):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['cases', 'temperature', 'humidity']])
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def train_lstm_model(X, y, epochs=50, batch_size=32):
    """Train LSTM model"""
    model = build_lstm_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Save model in new Keras format
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")
    
    return model

def load_lstm_model():
    """Load trained LSTM model"""
    # Try to load new format first, then fall back to old format
    if os.path.exists("models/lstm_model.keras"):
        return tf.keras.models.load_model("models/lstm_model.keras")
    elif os.path.exists("models/lstm_model.h5"):
        try:
            return tf.keras.models.load_model("models/lstm_model.h5")
        except (ValueError, OSError) as e:
            print(f"Warning: Could not load old model format: {e}")
            print("Please retrain the model.")
            return None
    return None