import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point

def preprocess_data(disease_df, climate_df, pop_df):
    """Merge and preprocess all data sources"""
    # Merge disease and climate data
    merged = pd.merge_asof(
        disease_df.sort_values('date'),
        climate_df.sort_values('date'),
        on='date',
        by='location',
        direction='nearest'
    )
    
    # Add coordinates for spatial join
    merged['latitude'] = 18.4655  # San Juan latitude
    merged['longitude'] = -66.1057  # San Juan longitude
    
    # Convert to GeoDataFrame
    merged = gpd.GeoDataFrame(
        merged, 
        geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
        crs="EPSG:4326"
    )
    
    # Spatial join with population data (using 'predicate' instead of 'op')
    merged = gpd.sjoin(merged, pop_df, how='left', predicate='within')
    
    # Use the correct location column (might be renamed during join)
    location_col = 'location_left' if 'location_left' in merged.columns else 'location'
    if location_col not in merged.columns:
        # If no location column exists, create one
        merged['location'] = 'San Juan'
        location_col = 'location'
    
    # Feature engineering
    merged['month'] = merged['date'].dt.month
    merged['week_of_year'] = merged['date'].dt.isocalendar().week
    merged['temp_humidity_index'] = merged['temperature'] * merged['humidity']
    merged['cases_lag_4w'] = merged.groupby(location_col)['cases'].shift(4)
    merged['rolling_avg_cases'] = merged.groupby(location_col)['cases'].transform(
        lambda x: x.rolling(window=4).mean()
    )
    
    # Handle missing data (using forward fill)
    merged = merged.ffill()
    
    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    merged.to_csv("data/processed/merged_data.csv", index=False)
    
    return merged