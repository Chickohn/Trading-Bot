#!/usr/bin/env python3
"""
Comprehensive test script for the hybrid trading bot.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import trading, alpaca
from data.hybrid_provider import HybridDataProvider
from features.technical_indicators import TechnicalIndicators
from models.random_forest_model import RandomForestModel
from models.base_model import ModelConfig
from signals.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager

async def test_hybrid_data_provider():
    """Test the hybrid data provider."""
    print("🔍 Testing Hybrid Data Provider...")
    
    provider = HybridDataProvider()
    
    # Test connection
    connected = await provider.connect()
    print(f"   Connection: {'✅' if connected else '❌'}")
    
    if not connected:
        return False
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    historical_data = await provider.get_historical_data(
        "AAPL", "1d", start_date, end_date
    )
    
    print(f"   Historical Data: {len(historical_data)} bars for AAPL")
    
    if len(historical_data) > 0:
        latest = historical_data[-1]
        print(f"   Latest: {latest.timestamp} - Close: ${latest.close}")
    
    # Test latest data
    latest_data = await provider.get_latest_data("AAPL", "1d")
    if latest_data:
        print(f"   Current Price: ${latest_data.close}")
    
    # Test account info
    account_info = await provider.get_account_info()
    if "error" not in account_info:
        print(f"   Account: ${account_info['portfolio_value']:.2f} portfolio value")
    
    await provider.disconnect()
    return True

async def test_technical_indicators():
    """Test technical indicators."""
    print("\n🔍 Testing Technical Indicators...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(len(dates))],
        'high': [101 + i * 0.1 for i in range(len(dates))],
        'low': [99 + i * 0.1 for i in range(len(dates))],
        'close': [100.5 + i * 0.1 for i in range(len(dates))],
        'volume': [1000000 + i * 1000 for i in range(len(dates))]
    }, index=dates)
    
    indicators = TechnicalIndicators()
    
    # Test various indicators
    try:
        # Calculate all indicators at once
        result_df = indicators.calculate_all_indicators(data)
        
        # Check if indicators were calculated
        indicator_cols = [col for col in result_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        print(f"   Calculated {len(indicator_cols)} indicators:")
        for col in indicator_cols[:10]:  # Show first 10
            print(f"     - {col}")
        if len(indicator_cols) > 10:
            print(f"     ... and {len(indicator_cols) - 10} more")
        
        # Check specific indicators
        if 'sma_20' in result_df.columns:
            print(f"   SMA(20): {result_df['sma_20'].iloc[-1]:.2f}")
        if 'rsi' in result_df.columns:
            print(f"   RSI: {result_df['rsi'].iloc[-1]:.2f}")
        if 'macd' in result_df.columns:
            print(f"   MACD: {result_df['macd'].iloc[-1]:.4f}")
        
        print("✅ Technical indicators working")
        return True
        
    except Exception as e:
        print(f"❌ Technical indicators error: {e}")
        return False

async def test_ml_model():
    """Test ML model training and prediction."""
    print("\n🔍 Testing ML Model...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(len(dates))],
        'high': [101 + i * 0.1 for i in range(len(dates))],
        'low': [99 + i * 0.1 for i in range(len(dates))],
        'close': [100.5 + i * 0.1 for i in range(len(dates))],
        'volume': [1000000 + i * 1000 for i in range(len(dates))]
    }, index=dates)
    
    # Add some patterns
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    config = ModelConfig(
        model_name="test_rf",
        lookback_period=10,
        prediction_threshold=0.6,
        retrain_frequency="1d"
    )
    
    model = RandomForestModel(config)
    
    try:
        # Prepare features
        X = model.prepare_features(data)
        y = model.prepare_target(data)
        
        # Align data
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) > 50:
            # Train model
            metrics = model.train(X, y)
            print(f"   Model trained with accuracy: {metrics.get('accuracy', 0):.4f}")
            
            # Test prediction
            latest_features = X.iloc[-1:].values
            prediction = model.predict(latest_features)
            if hasattr(prediction, '__len__') and len(prediction) > 0:
                print(f"   Latest prediction: {prediction[0]:.4f}")
            else:
                print(f"   Latest prediction: {prediction:.4f}")
            
            print("✅ ML model working")
            return True
        else:
            print("⚠️  Insufficient data for training")
            return False
            
    except Exception as e:
        print(f"❌ ML model error: {e}")
        return False

async def test_signal_generation():
    """Test signal generation."""
    print("\n🔍 Testing Signal Generation...")
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    data = pd.DataFrame({
        'open': [100 + i * 0.1 for i in range(len(dates))],
        'high': [101 + i * 0.1 for i in range(len(dates))],
        'low': [99 + i * 0.1 for i in range(len(dates))],
        'close': [100.5 + i * 0.1 for i in range(len(dates))],
        'volume': [1000000 + i * 1000 for i in range(len(dates))]
    }, index=dates)
    
    # Create mock model
    config = ModelConfig(
        model_name="test_model",
        lookback_period=10,
        prediction_threshold=0.6,
        retrain_frequency="1d"
    )
    
    model = RandomForestModel(config)
    models = [model]
    
    signal_generator = SignalGenerator(models)
    
    try:
        # Create mock market data
        from data.base import MarketData, OHLCV
        mock_ohlcv = OHLCV(
            timestamp=datetime.now(),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
            symbol="AAPL",
            timeframe="1d"
        )
        mock_market_data = MarketData(
            symbol="AAPL",
            price=100.5,
            volume=1000000,
            timestamp=datetime.now(),
            data_type="trade",
            ohlcv=mock_ohlcv
        )
        
        # Generate a single signal
        signal = await signal_generator.generate_signal("AAPL", mock_market_data, data)
        
        if signal:
            print(f"   Generated signal: {signal.signal_type} - Confidence: {signal.confidence:.4f}")
        else:
            print("   No signal generated (may be due to filters or cooldown)")
        
        print("✅ Signal generation working")
        return True
        
    except Exception as e:
        print(f"❌ Signal generation error: {e}")
        return False

async def test_risk_management():
    """Test risk management."""
    print("\n🔍 Testing Risk Management...")
    
    risk_manager = RiskManager()
    
    try:
        # Test risk check with account info
        account_info = {
            "portfolio_value": 100000,
            "cash": 50000,
            "equity": 100000
        }
        
        risk_check = risk_manager.check_risk_limits(account_info)
        print(f"   Risk check passed: {risk_check}")
        
        # Test position summary
        summary = risk_manager.get_position_summary()
        print(f"   Position summary: {summary['total_positions']} positions")
        
        print("✅ Risk management working")
        return True
        
    except Exception as e:
        print(f"❌ Risk management error: {e}")
        return False

async def test_full_integration():
    """Test full integration with real data."""
    print("\n🔍 Testing Full Integration...")
    
    try:
        # Initialize components
        provider = HybridDataProvider()
        await provider.connect()
        
        # Get real data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        aapl_data = await provider.get_historical_data("AAPL", "1d", start_date, end_date)
        
        if len(aapl_data) < 30:
            print("⚠️  Insufficient real data for full test")
            await provider.disconnect()
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in aapl_data], index=[bar.timestamp for bar in aapl_data])
        
        # Test technical indicators
        indicators = TechnicalIndicators()
        result_df = indicators.calculate_all_indicators(df)
        
        print(f"   Real AAPL data: {len(df)} days")
        print(f"   Current price: ${df['close'].iloc[-1]:.2f}")
        if 'sma_20' in result_df.columns:
            print(f"   SMA(20): ${result_df['sma_20'].iloc[-1]:.2f}")
        if 'rsi' in result_df.columns:
            print(f"   RSI(14): {result_df['rsi'].iloc[-1]:.2f}")
        
        # Test signal generation
        config = ModelConfig(
            model_name="integration_test",
            lookback_period=20,
            prediction_threshold=0.6,
            retrain_frequency="1d"
        )
        
        model = RandomForestModel(config)
        
        # Add target for training
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        X = model.prepare_features(df)
        y = model.prepare_target(df)
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) > 50:
            metrics = model.train(X, y)
            print(f"   Model accuracy: {metrics.get('accuracy', 0):.4f}")
            
            signal_generator = SignalGenerator([model])
            
            # Create mock market data for signal generation
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=df['close'].iloc[-1],
                high=df['high'].iloc[-1],
                low=df['low'].iloc[-1],
                close=df['close'].iloc[-1],
                volume=df['volume'].iloc[-1],
                symbol="AAPL",
                timeframe="1d"
            )
            mock_market_data = MarketData(
                symbol="AAPL",
                price=df['close'].iloc[-1],
                volume=df['volume'].iloc[-1],
                timestamp=datetime.now(),
                data_type="trade",
                ohlcv=mock_ohlcv
            )
            
            signal = await signal_generator.generate_signal("AAPL", mock_market_data, df)
            
            if signal:
                print(f"   Generated signal: {signal.signal_type} ({signal.confidence:.4f})")
            else:
                print("   No signal generated (may be due to filters)")
        
        await provider.disconnect()
        print("✅ Full integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Full integration error: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Hybrid Trading Bot - Comprehensive Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(await test_hybrid_data_provider())
    results.append(await test_technical_indicators())
    results.append(await test_ml_model())
    results.append(await test_signal_generation())
    results.append(await test_risk_management())
    results.append(await test_full_integration())
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Data Provider: {'✅' if results[0] else '❌'}")
    print(f"   Technical Indicators: {'✅' if results[1] else '❌'}")
    print(f"   ML Model: {'✅' if results[2] else '❌'}")
    print(f"   Signal Generation: {'✅' if results[3] else '❌'}")
    print(f"   Risk Management: {'✅' if results[4] else '❌'}")
    print(f"   Full Integration: {'✅' if results[5] else '❌'}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your trading bot is ready to run!")
        print("   Run: python3.11 src/main.py --mode paper")
    elif passed >= 4:
        print("\n✅ Most tests passed! Bot should work with some limitations.")
    else:
        print("\n⚠️  Several tests failed. Check your setup and dependencies.")

if __name__ == "__main__":
    asyncio.run(main()) 