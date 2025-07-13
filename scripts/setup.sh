#!/bin/bash

# Trading Bot Setup Script
set -e

echo "🚀 Setting up Advanced Trading Bot System..."

# Check if Python 3.11+ is installed
python_version=$(python3.11 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Current version: $python_version"
    echo "💡 Try: brew install python@3.11"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python3.11 -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
python3.11 -m pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs data notebooks configs

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
else
    echo "✅ .env file already exists"
fi

# Setup database (if using local PostgreSQL)
if command -v psql &> /dev/null; then
    echo "🗄️  Setting up database..."
    # This would create the database and tables
    # psql -U postgres -c "CREATE DATABASE trading_bot;"
    echo "ℹ️  Database setup requires manual configuration"
else
    echo "ℹ️  PostgreSQL not found. Using Docker for database."
fi

# Setup MLflow
echo "🔬 Setting up MLflow..."
mkdir -p mlruns

# Create initial configuration
echo "📝 Creating initial configuration..."
cat > configs/trading_config.yaml << EOF
# Trading Bot Configuration
trading:
  mode: paper
  symbols: ["AAPL", "TSLA", "MSFT"]
  timeframe: "1m"
  max_positions: 5

risk:
  max_position_size: 0.1
  max_daily_loss: 0.05
  stop_loss_pct: 0.02
  take_profit_pct: 0.04

models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    prediction_threshold: 0.7
EOF

# Setup monitoring
echo "📊 Setting up monitoring..."
mkdir -p docker/grafana/dashboards docker/grafana/datasources

# Create Prometheus configuration
cat > docker/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['trading-bot:8000']
    metrics_path: '/metrics'
EOF

# Create Grafana datasource
mkdir -p docker/grafana/datasources
cat > docker/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Setup Jupyter notebooks
echo "📓 Setting up Jupyter notebooks..."
mkdir -p notebooks
cat > notebooks/01_data_analysis.ipynb << EOF
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Trading Bot Data Analysis\\n",
    "\\n",
    "This notebook provides tools for analyzing market data and model performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "# Import trading bot modules\\n",
    "import sys\\n",
    "sys.path.append('..')\\n",
    "\\n",
    "from src.data.alpaca_provider import AlpacaDataProvider\\n",
    "from src.features.technical_indicators import TechnicalIndicators"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create test script
echo "🧪 Creating test script..."
cat > scripts/test_setup.py << EOF
#!/usr/bin/env python3
"""Test script to verify trading bot setup."""

import sys
import os

# Ensure src is on the Python path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

def test_imports():
    """Test if all modules can be imported."""
    try:
        from src.utils.config import config
        print("✅ Configuration module imported successfully")
        
        from src.data.alpaca_provider import AlpacaDataProvider
        print("✅ Alpaca provider imported successfully")
        
        from src.features.technical_indicators import TechnicalIndicators
        print("✅ Technical indicators imported successfully")
        
        from src.models.random_forest_model import RandomForestModel
        print("✅ Random Forest model imported successfully")
        
        from src.signals.signal_generator import SignalGenerator
        print("✅ Signal generator imported successfully")
        
        from src.risk.risk_manager import RiskManager
        print("✅ Risk manager imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    try:
        from src.utils.config import config
        print(f"✅ Configuration loaded: {config.environment}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing trading bot setup...")
    
    success = True
    success &= test_imports()
    success &= test_configuration()
    
    if success:
        print("\\n🎉 All tests passed! Trading bot is ready to use.")
    else:
        print("\\n❌ Some tests failed. Please check the setup.")
        sys.exit(1)
EOF

# Make scripts executable
chmod +x scripts/test_setup.py

# Run tests
echo "🧪 Running setup tests..."
python3.11 scripts/test_setup.py

# Create startup script
echo "🚀 Creating startup script..."
cat > scripts/start.sh << 'EOF'
#!/bin/bash

# Trading Bot Startup Script

# Activate virtual environment
source venv/bin/activate

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

# Start the trading bot
echo "🚀 Starting trading bot..."
python3.11 -m src.main --mode paper
EOF

chmod +x scripts/start.sh

# Create Docker startup script
cat > scripts/start_docker.sh << 'EOF'
#!/bin/bash

# Docker Trading Bot Startup Script

echo "🐳 Starting trading bot with Docker Compose..."

# Build and start services
docker-compose up -d

echo "✅ Services started. Access points:"
echo "   - Trading Bot: http://localhost:8000"
echo "   - MLflow: http://localhost:5000"
echo "   - Grafana: http://localhost:3000 (admin/admin)"
echo "   - Prometheus: http://localhost:9090"
echo "   - Jupyter: http://localhost:8888"
echo "   - Flower: http://localhost:5555 (admin/admin)"

# Show logs
echo "📋 Showing logs (Ctrl+C to stop):"
docker-compose logs -f trading-bot
EOF

chmod +x scripts/start_docker.sh

echo ""
echo "🎉 Trading Bot setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: ./scripts/start.sh (for local development)"
echo "3. Run: ./scripts/start_docker.sh (for Docker deployment)"
echo ""
echo "📚 Documentation: README.md"
echo "🧪 Test setup: python3.11 scripts/test_setup.py"
echo "" 