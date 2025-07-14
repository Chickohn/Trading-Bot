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
        print("\n🎉 All tests passed! Trading bot is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        sys.exit(1)
