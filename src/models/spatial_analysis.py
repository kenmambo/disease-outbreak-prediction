from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os

def spatial_feature_importance(data):
    """Train Random Forest and get feature importance"""
    rf = RandomForestRegressor(n_estimators=100)
    features = ['temperature', 'humidity', 'population_density', 'cases_lag_4w']
    X = data[features]
    y = data['cases']
    rf.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_model.pkl")
    
    importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf, importance

def load_rf_model():
    """Load trained Random Forest model"""
    if os.path.exists("models/rf_model.pkl"):
        return joblib.load("models/rf_model.pkl")
    return None