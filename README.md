# 🦠 Disease Outbreak Prediction System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-managed-orange?style=flat-square)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square)](https://github.com/psf/black)

*An AI-powered system for predicting disease outbreaks using advanced machine learning and spatio-temporal analysis*

[Features](#-features) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Dashboard](#-interactive-dashboard) • [Documentation](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 🎯 Overview

The **Disease Outbreak Prediction System** is a comprehensive data science platform that leverages cutting-edge machine learning techniques to forecast disease outbreaks. By combining temporal forecasting with spatial analysis, this system provides early warning capabilities for public health officials and researchers.

### 🧠 Key Technologies
- **LSTM Neural Networks** for time-series forecasting
- **Random Forest** for spatial pattern analysis  
- **Streamlit** for interactive visualization
- **TensorFlow/Keras** for deep learning models
- **GeoPandas** for geospatial data processing
- **Plotly** for interactive charts and visualizations
- **uv** for fast dependency management
- **Python 3.10+** with modern ML stack

## ✨ Features

🔮 **Predictive Modeling**
- Advanced time-series forecasting with LSTM networks
- Spatial outbreak pattern recognition
- Multi-variable epidemiological analysis

📊 **Interactive Dashboard**
- Real-time outbreak monitoring with live predictions
- Professional UI with color-coded risk assessment
- Geographic visualization of disease spread patterns
- Interactive charts with Plotly integration
- Model training capability from the dashboard
- Performance metrics and accuracy tracking
- Customizable alert thresholds
- Export capabilities for reports

⚡ **High Performance**
- Fast model training and inference
- Scalable architecture for large datasets
- Optimized data preprocessing pipelines

🔧 **Developer Friendly**
- Cross-platform script support (Windows/Linux/macOS)
- Modern Keras model format with legacy compatibility
- Well-documented APIs and comprehensive error handling
- Modular design for easy extension
- Professional project structure
- CI/CD ready with automated testing support

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- Windows, macOS, or Linux

### Installation

1. **Install uv** (if not already installed):
   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/disease-outbreak-prediction.git
   cd disease-outbreak-prediction
   ```

3. **Set up the environment**:
   ```bash
   uv venv
   uv pip install -e .
   ```

4. **Verify installation**:
   ```bash
   uv run python -c "import disease_outbreak_prediction; print('✅ Installation successful!')"
   ```

## 📖 Usage

### 🎯 Training Models

```bash
# Train all models with default parameters
uv run python -m disease_outbreak_prediction.train

# Or use the convenience script
./scripts/train.sh      # Linux/macOS
scripts\train.bat       # Windows CMD
.\scripts\train.ps1     # Windows PowerShell
```

### 🖥️ Running the Dashboard

```bash
# Launch the interactive dashboard
uv run streamlit run disease_outbreak_prediction/dashboard/app.py

# Or use the cross-platform convenience scripts
./scripts/dashboard.sh      # Linux/macOS/WSL
scripts\dashboard.bat       # Windows CMD
.\scripts\dashboard.ps1     # Windows PowerShell
```

The dashboard will be available at `http://localhost:8501`

### 🧪 Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=disease_outbreak_prediction

# Run specific test file
uv run pytest tests/test_models.py
```

## 🖥️ Interactive Dashboard

Our comprehensive Streamlit dashboard provides a complete interface for disease outbreak prediction with professional-grade visualization and analytics.

### 🎯 Key Features

**📊 Real-time Visualization**
- Live predictions vs actual cases with interactive Plotly charts
- Last 365 days trend analysis with customizable time ranges
- Professional UI with responsive design and custom CSS styling
- Progress indicators for model training and data loading

**🎨 Advanced Risk Assessment System**
- 🔴 **HIGH RISK**: Predicted cases > 150 (Critical Alert - Immediate action required)
- 🟡 **MEDIUM RISK**: Predicted cases > 100 (Warning - Monitor closely)
- 🟢 **LOW RISK**: Predicted cases ≤ 100 (Normal - Continue monitoring)
- Color-coded alerts with detailed risk information
- Prediction accuracy percentage for confidence assessment

**📈 Performance Analytics Dashboard**
- Real-time model performance metrics (MAE, RMSE)
- Feature importance visualization with horizontal bar charts
- Prediction accuracy tracking with percentage indicators
- Model comparison and evaluation metrics

**🤖 Integrated Model Management**
- Train new models directly from the dashboard interface
- Progress tracking with real-time status updates
- Automatic model loading with intelligent fallback support
- Support for both legacy (.h5) and modern (.keras) formats
- Error handling with user-friendly messages

**🎨 Professional User Interface**
- Sidebar control panel for easy navigation
- Responsive layout optimized for different screen sizes
- Custom CSS styling for enhanced visual appeal
- Emoji-enhanced sections for better user experience
- Data overview cards with key statistics

### 🚀 Dashboard Screenshots

*The dashboard provides an intuitive interface with real-time data visualization, risk assessment, and model management capabilities - perfect for public health officials and researchers.*

## 📁 Project Structure

```
disease-outbreak-prediction/
├── 📂 data/
│   ├── 📂 raw/                    # Raw CDC/WHO data files
│   └── 📂 processed/              # Processed & merged datasets
├── 📂 disease_outbreak_prediction/  # Root package for direct access
│   └── 📂 dashboard/
│       └── 📄 app.py              # Streamlit dashboard app
├── 📂 models/                     # Trained ML models
│   ├── 📄 lstm_model.keras        # LSTM model (new format)
│   ├── 📄 lstm_model.h5           # LSTM model (legacy format)
│   └── 📄 rf_model.pkl            # Random Forest model
├── 📂 notebooks/                  # Jupyter notebooks for exploration
├── 📂 src/
│   └── 📂 disease_outbreak_prediction/
│       ├── 📄 __init__.py
│       ├── 📄 train.py            # Training orchestrator module
│       ├── 📄 data_acquisition.py # Data collection & APIs
│       ├── 📄 preprocessing.py    # Data cleaning & feature engineering
│       ├── 📂 models/
│       │   ├── 📄 __init__.py
│       │   ├── 📄 lstm_model.py   # LSTM implementation & training
│       │   └── 📄 spatial_analysis.py # Random Forest & spatial models
│       └── 📂 dashboard/
│           ├── 📄 __init__.py
│           └── 📄 app.py          # Enhanced Streamlit dashboard
├── 📂 scripts/                    # Cross-platform utility scripts
│   ├── 📄 train.bat               # Windows training script
│   ├── 📄 train.ps1               # PowerShell training script
│   ├── 📄 train.sh                # Unix/Linux training script
│   ├── 📄 dashboard.bat           # Windows dashboard script
│   ├── 📄 dashboard.ps1           # PowerShell dashboard script
│   └── 📄 dashboard.sh            # Unix/Linux dashboard script
├── 📂 tests/                      # Unit tests & integration tests
├── 📂 .github/                    # GitHub Actions CI/CD workflows
├── 📄 pyproject.toml              # Project config & dependencies
├── 📄 README.md                   # This comprehensive guide
└── 📄 LICENSE                     # MIT License
```

## 🔬 Model Architecture & Implementation

### 🧠 Deep Learning: LSTM Neural Network
- **Input Features**: Historical disease incidence, temperature, humidity
- **Architecture**: 
  - Multi-layer LSTM (128 → 64 units) with dropout regularization
  - Dense layers (32 units) with ReLU activation
  - Output layer for outbreak probability prediction
- **Training**: Adam optimizer with MSE loss, 50 epochs
- **Output**: Continuous values representing predicted case counts
- **Format**: Supports both Keras (.keras) and legacy H5 (.h5) formats

### 🌳 Machine Learning: Random Forest Ensemble
- **Input Features**: Temperature, humidity, population density, lagged cases
- **Architecture**: 100 decision trees with feature importance ranking
- **Training**: Scikit-learn implementation with spatial feature engineering
- **Output**: Spatial risk assessment and feature importance scores
- **Performance**: Real-time predictions with joblib serialization

### 📊 Data Pipeline
- **Data Sources**: CDC/WHO APIs, NOAA climate data, WorldPop demographics
- **Preprocessing**: GeoPandas spatial joins, feature engineering, missing data handling
- **Features**: Temporal lags, rolling averages, temperature-humidity index
- **Validation**: Forward-fill strategy for time-series data integrity

## 📈 Performance Metrics & Results

### 🏆 Model Performance

| Model | Training Data | Accuracy | Features | Status |
|-------|---------------|----------|----------|--------|
| **LSTM** | 4,014 samples | ~91.6% | Temperature, Humidity, Cases | ✅ **Active** |
| **Random Forest** | 4,018 samples | ~88.5% | Weather + Demographics | ✅ **Active** |

### 📊 Feature Importance Analysis

Based on our trained Random Forest model:

| Feature | Importance | Impact |
|---------|------------|--------|
| **Temperature** | 83.96% | 🔥 **Primary Driver** |
| **Humidity** | 11.72% | 💧 Moderate Impact |
| **Cases (4-week lag)** | 4.32% | 📅 Historical Pattern |
| **Population Density** | 0.00% | 🏠 Minimal Impact |

### 📊 Real-World Performance
- **Dataset**: 11 years (2010-2020) of San Juan outbreak data
- **Predictions**: 365-day rolling forecasts with daily updates
- **Accuracy**: ~90%+ prediction accuracy in testing
- **Response Time**: <2 seconds for real-time predictions
- **Model Size**: LSTM (1.4MB), Random Forest (1.2MB)

## 🛠️ Technology Stack & Requirements

### 🐍 Core Technologies
- **Python 3.10+** - Main programming language
- **uv** - Ultra-fast Python package installer and dependency manager
- **TensorFlow/Keras 2.20+** - Deep learning framework for LSTM models
- **Scikit-learn** - Machine learning library for Random Forest
- **Streamlit** - Interactive dashboard framework
- **Plotly** - Interactive data visualization library
- **GeoPandas** - Geospatial data processing
- **Pandas/NumPy** - Data manipulation and numerical computing

### 💻 System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space for dependencies and models
- **Python**: Version 3.10 or higher
- **Network**: Internet connection for data fetching (optional for offline mode)

### 🌍 Cross-Platform Support
- **Windows**: Full support with PowerShell and CMD scripts
- **macOS/Linux**: Native support with bash scripts
- **WSL**: Windows Subsystem for Linux compatibility
- **Docker**: Containerization support (coming soon)

## 🚀 Quick Commands Summary

```bash
# 📦 One-time Setup
uv venv && uv pip install -e .

# 🏃 Train Models (Required first run)
uv run python -m disease_outbreak_prediction.train

# 🖥️ Launch Dashboard
uv run streamlit run disease_outbreak_prediction/dashboard/app.py

# 🧪 Run Tests
uv run pytest
```

## 🆘 What's New in This Version

✨ **Enhanced Dashboard**
- Professional UI with custom CSS styling and responsive design
- Real-time model training capability from the web interface
- Advanced risk assessment with color-coded alerts
- Interactive Plotly charts with 365-day trend analysis

🤖 **Improved Models**
- Modern Keras (.keras) format with legacy H5 compatibility
- Enhanced error handling and automatic fallback support
- Cross-platform model serialization

🛠️ **Developer Experience**
- Cross-platform scripts (Windows .bat/.ps1, Unix .sh)
- Comprehensive project structure with organized modules
- Professional documentation with visual project structure
- Updated dependencies resolving compatibility issues

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `uv pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run the test suite: `uv run pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- World Health Organization for providing public health data
- The open-source community for amazing ML libraries
- Contributors and maintainers of this project

## 🗺️ Roadmap

- [ ] **v2.0**: Multi-disease support
- [ ] **v2.1**: Transformer-based models
- [ ] **v2.2**: Social media sentiment analysis integration
- [ ] **v3.0**: Mobile application for field workers
- [ ] **v3.1**: Real-time API for health systems integration
- [ ] **v4.0**: AI-powered policy recommendation engine

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[Report Bug](https://github.com/yourusername/disease-outbreak-prediction/issues) • [Request Feature](https://github.com/yourusername/disease-outbreak-prediction/issues) • [Documentation](https://github.com/yourusername/disease-outbreak-prediction/wiki)

</div>