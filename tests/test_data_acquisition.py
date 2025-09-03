import pytest
import pandas as pd
from disease_outbreak_prediction.data_acquisition import fetch_disease_data, fetch_climate_data, get_population_data

def test_fetch_disease_data():
    data = fetch_disease_data()
    assert isinstance(data, pd.DataFrame)
    assert 'date' in data.columns
    assert 'cases' in data.columns

def test_fetch_climate_data():
    data = fetch_climate_data("San Juan", "2020-01-01", "2020-01-31")
    assert isinstance(data, pd.DataFrame)
    assert 'date' in data.columns
    assert 'temperature' in data.columns

def test_get_population_data():
    data = get_population_data()
    assert 'population_density' in data.columns
    assert 'geometry' in data.columns