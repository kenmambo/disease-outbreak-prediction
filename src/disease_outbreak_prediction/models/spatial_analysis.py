from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os

def spatial_feature_importance(data):
    """Train Random Forest and get feature importance"""
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Check which features are available in the data
    available_features = []
    potential_features = ['temperature', 'humidity', 'population_density', 'cases_lag_4w']
    
    for feature in potential_features:
        if feature in data.columns:
            available_features.append(feature)
        else:
            print(f"Warning: Feature '{feature}' not found in data. Skipping.")
    
    if not available_features:
        print("No suitable features found. Using available numeric columns.")
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if 'cases' in numeric_cols:
            numeric_cols.remove('cases')
        available_features = numeric_cols[:4]  # Take first 4 numeric columns
    
    X = data[available_features]
    y = data['cases']
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    rf.fit(X, y)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/rf_model.pkl")
    
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf, importance

def load_rf_model():
    """Load trained Random Forest model"""
    if os.path.exists("models/rf_model.pkl"):
        return joblib.load("models/rf_model.pkl")
    return None