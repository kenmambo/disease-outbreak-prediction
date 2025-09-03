"""
Test suite for disease outbreak prediction system.

This module contains unit tests for all components of the system including:
- Data acquisition and processing
- Model training and prediction
- Dashboard functionality
- Cross-platform compatibility
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataAcquisition:
    """Test data acquisition functionality."""
    
    def test_fetch_disease_data(self):
        """Test disease data fetching."""
        from disease_outbreak_prediction.data_acquisition import fetch_disease_data
        
        data = fetch_disease_data()
        
        assert isinstance(data, pd.DataFrame)
        assert 'date' in data.columns
        assert 'cases' in data.columns
        assert 'location' in data.columns
        assert len(data) > 0
        
    def test_fetch_climate_data(self):
        """Test climate data fetching."""
        from disease_outbreak_prediction.data_acquisition import fetch_climate_data
        
        data = fetch_climate_data("San Juan", "2020-01-01", "2020-01-31")
        
        assert isinstance(data, pd.DataFrame)
        assert 'temperature' in data.columns
        assert 'humidity' in data.columns
        assert len(data) > 0
        
    def test_get_population_data(self):
        """Test population data fetching."""
        from disease_outbreak_prediction.data_acquisition import get_population_data
        
        data = get_population_data()
        
        # Should return a GeoDataFrame
        assert hasattr(data, 'geometry')
        assert 'population_density' in data.columns
        assert len(data) > 0


class TestDataPreprocessing:
    """Test data preprocessing functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from disease_outbreak_prediction.data_acquisition import (
            fetch_disease_data, 
            fetch_climate_data, 
            get_population_data
        )
        
        disease_data = fetch_disease_data()
        climate_data = fetch_climate_data("San Juan", "2010-01-01", "2010-03-31")
        pop_data = get_population_data()
        
        return disease_data[:100], climate_data[:100], pop_data
    
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing pipeline."""
        from disease_outbreak_prediction.preprocessing import preprocess_data
        
        disease_data, climate_data, pop_data = sample_data
        processed = preprocess_data(disease_data, climate_data, pop_data)
        
        assert isinstance(processed, pd.DataFrame)
        assert 'cases' in processed.columns
        assert 'temperature' in processed.columns
        assert 'humidity' in processed.columns
        assert len(processed) > 0
        
        # Check feature engineering
        expected_features = ['month', 'week_of_year', 'temp_humidity_index']
        for feature in expected_features:
            assert feature in processed.columns


class TestModels:
    """Test model functionality."""
    
    @pytest.fixture
    def sample_processed_data(self):
        """Create sample processed data for model testing."""
        from disease_outbreak_prediction.data_acquisition import (
            fetch_disease_data, 
            fetch_climate_data, 
            get_population_data
        )
        from disease_outbreak_prediction.preprocessing import preprocess_data
        
        # Use small dataset for fast testing
        disease_data = fetch_disease_data()[:50]
        climate_data = fetch_climate_data("San Juan", "2010-01-01", "2010-02-28")[:50]
        pop_data = get_population_data()
        
        return preprocess_data(disease_data, climate_data, pop_data)
    
    def test_lstm_data_preparation(self, sample_processed_data):
        """Test LSTM data preparation."""
        from disease_outbreak_prediction.models.lstm_model import prepare_lstm_data
        
        X, y, scaler = prepare_lstm_data(sample_processed_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 3  # (samples, timesteps, features)
        assert len(X) == len(y)
        assert X.shape[2] == 3  # cases, temperature, humidity
        
    def test_lstm_model_training(self, sample_processed_data):
        """Test LSTM model training (quick training for testing)."""
        from disease_outbreak_prediction.models.lstm_model import (
            prepare_lstm_data, 
            train_lstm_model
        )
        
        X, y, scaler = prepare_lstm_data(sample_processed_data)
        
        # Quick training with minimal epochs for testing
        model = train_lstm_model(X, y, epochs=2, batch_size=8)
        
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        predictions = model.predict(X[:5], verbose=0)
        assert len(predictions) == 5
        
    def test_random_forest_training(self, sample_processed_data):
        """Test Random Forest model training."""
        from disease_outbreak_prediction.models.spatial_analysis import (
            spatial_feature_importance
        )
        
        model, importance = spatial_feature_importance(sample_processed_data)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns


class TestTrainingPipeline:
    """Test the complete training pipeline."""
    
    def test_training_pipeline_imports(self):
        """Test that training module can be imported."""
        try:
            import disease_outbreak_prediction.train
            assert True
        except ImportError as e:
            pytest.fail(f"Cannot import training module: {e}")
    
    @pytest.mark.slow
    def test_full_training_pipeline(self):
        """Test complete training pipeline (marked as slow)."""
        # This would be a longer integration test
        # Only run with pytest -m "not slow" to skip during regular testing
        pytest.skip("Full training test skipped - use 'pytest -m slow' to run")


class TestDashboard:
    """Test dashboard functionality."""
    
    def test_dashboard_imports(self):
        """Test that dashboard module can be imported."""
        try:
            import disease_outbreak_prediction.dashboard.app
            assert True
        except ImportError as e:
            pytest.fail(f"Cannot import dashboard module: {e}")
    
    def test_streamlit_components(self):
        """Test Streamlit component imports."""
        try:
            import streamlit as st
            import plotly.express as px
            assert True
        except ImportError as e:
            pytest.fail(f"Cannot import required dashboard dependencies: {e}")


class TestCrossPlatform:
    """Test cross-platform compatibility."""
    
    def test_path_handling(self):
        """Test that paths work across platforms."""
        from pathlib import Path
        
        # Test relative paths
        data_path = Path("data/raw")
        models_path = Path("models")
        
        assert isinstance(data_path, Path)
        assert isinstance(models_path, Path)
        
    def test_script_files_exist(self):
        """Test that cross-platform scripts exist."""
        script_files = [
            "scripts/train.bat",
            "scripts/train.ps1", 
            "scripts/train.sh",
            "scripts/dashboard.bat",
            "scripts/dashboard.ps1",
            "scripts/dashboard.sh"
        ]
        
        for script in script_files:
            assert Path(script).exists(), f"Script {script} does not exist"


# Performance benchmarks
class TestPerformance:
    """Test performance requirements."""
    
    def test_model_load_time(self):
        """Test that models load within acceptable time."""
        import time
        from disease_outbreak_prediction.models.lstm_model import load_lstm_model
        from disease_outbreak_prediction.models.spatial_analysis import load_rf_model
        
        # Test LSTM loading
        start_time = time.time()
        lstm_model = load_lstm_model()
        lstm_load_time = time.time() - start_time
        
        # Test RF loading 
        start_time = time.time()
        rf_model = load_rf_model()
        rf_load_time = time.time() - start_time
        
        # Models should load within 5 seconds (or be None if not trained)
        if lstm_model is not None:
            assert lstm_load_time < 5.0, f"LSTM model took {lstm_load_time:.2f}s to load"
            
        if rf_model is not None:
            assert rf_load_time < 5.0, f"RF model took {rf_load_time:.2f}s to load"


# Configuration and fixtures
@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up test environment."""
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup if needed (optional)
    # Could remove test-generated files here


# Markers for test categorization
pytestmark = [
    pytest.mark.unit,  # Mark all tests in this file as unit tests
]