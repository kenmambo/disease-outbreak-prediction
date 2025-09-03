import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import os
from pathlib import Path

def fetch_disease_data():
    """Fetch disease data from CDC or WHO API"""
    # For demo purposes, we'll create sample data
    # In production, replace with actual API calls
    data_path = Path("data/raw/cdc_flu_data.csv")
    
    if not data_path.exists():
        # Create sample data
        date_range = pd.date_range(start="2010-01-01", end="2020-12-31")
        disease_df = pd.DataFrame({
            'date': date_range,
            'location': ['San Juan'] * len(date_range),
            'cases': [int(50 + 30 * (i % 12) + 10 * (i % 7)) for i in range(len(date_range))]
        })
        disease_df.to_csv(data_path, index=False)
    else:
        disease_df = pd.read_csv(data_path)
    
    disease_df['date'] = pd.to_datetime(disease_df['date'])
    return disease_df

def fetch_climate_data(location, start_date, end_date):
    """Fetch climate data from NOAA API"""
    # For demo purposes, we'll create sample data
    # In production, replace with actual NOAA API calls
    date_range = pd.date_range(start=start_date, end=end_date)
    climate_df = pd.DataFrame({
        'date': date_range,
        'location': [location] * len(date_range),
        'temperature': [25 + 5 * (i % 12) / 12 for i in range(len(date_range))],
        'humidity': [70 + 10 * (i % 6) / 6 for i in range(len(date_range))],
        'precipitation': [5 * (i % 4) for i in range(len(date_range))]
    })
    return climate_df

def get_population_data():
    """Get population density data"""
    # For demo purposes, we'll create sample data
    # In production, replace with actual WorldPop data
    data = {
        'location': ['San Juan'],
        'population_density': [1000],
        'geometry': [Point(-66.1057, 18.4655)]  # San Juan coordinates
    }
    return gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")