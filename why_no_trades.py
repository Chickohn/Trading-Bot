#!/usr/bin/env python3
"""
Why No Trades? - Analyze why your trading bot hasn't made trades yet.
"""

import sys
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.hybrid_provider import HybridDataProvider
from features.technical_indicators import TechnicalIndicators
from models.random_forest_model import RandomForestModel
from models.base_model import ModelConfig
from signals.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager
from data.base import MarketData, OHLCV
from utils.config import trading, risk

async def analyze_why_no_trades():
    """Analyze why the bot hasn't made trades yet."""
    print("🤔 WHY NO TRADES? - Trading Bot Analysis")
    print("=" * 60)
    
    # Initialize components
    provider = HybridDataProvider()
    indicators = TechnicalIndicators()
    risk_manager = RiskManager()
    
    await provider.connect()
    
    # Get account info
    account_info = await provider.get_account_info()
    print(f"💰 Account Status:")
    print(f"   Portfolio Value: ${account_info.get('portfolio_value', 0):,.2f}")
    print(f"   Available Cash: ${account_info.get('cash', 0):,.2f}")
    print(f"   Buying Power: ${account_info.get('buying_power', 0):,.2f}")
    
    # Check configuration
    print(f"\n⚙️  Trading Configuration:")
    print(f"   Symbols: {trading.trading_symbols}")
    print(f"   Default Timeframe: {trading.default_timeframe}")
    print(f"   Signal Cooldown: {trading.signal_cooldown} seconds")
    print(f"   Pattern Threshold: {trading.pattern_threshold}")
    
    print(f"\n🛡️  Risk Management Settings:")
    print(f"   Max Position Size: {risk.max_position_size:.1%} of portfolio")
    print(f"   Max Daily Loss: {risk.max_daily_loss:.1%}")
    print(f"   Stop Loss: {risk.stop_loss_pct:.1%}")
    print(f"   Take Profit: {risk.take_profit_pct:.1%}")
    print(f"   Max Open Trades: {risk.max_open_trades}")
    
    # Analyze each symbol
    print(f"\n🔍 Symbol Analysis:")
    print("=" * 60)
    
    # Train a model for analysis
    config = ModelConfig(
        model_name="analysis",
        lookback_period=20,
        prediction_threshold=0.7,
        retrain_frequency="1d"
    )
    model = RandomForestModel(config)
    
    # Train with AAPL data
    print("🎓 Training analysis model...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    training_data = await provider.get_historical_data("AAPL", "1d", start_date, end_date)
    if training_data and len(training_data) > 100:
        df = pd.DataFrame([{
            'open': bar.open, 'high': bar.high, 'low': bar.low,
            'close': bar.close, 'volume': bar.volume
        } for bar in training_data])
        
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        X = model.prepare_features(df)
        y = model.prepare_target(df)
        
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) > 50:
            metrics = model.train(X, y)
            print(f"✅ Model trained with {metrics.get('accuracy', 0):.1%} accuracy")
    
    signal_generator = SignalGenerator([model])
    
    reasons_no_trades = []
    
    for symbol in trading.trading_symbols[:3]:  # Check first 3 symbols
        print(f"\n📊 Analyzing {symbol}...")
        
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            historical_data = await provider.get_historical_data(symbol, "1d", start_date, end_date)
            
            if not historical_data or len(historical_data) < 20:
                print(f"   ❌ Insufficient data ({len(historical_data) if historical_data else 0} bars)")
                reasons_no_trades.append(f"{symbol}: Insufficient historical data")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': bar.open, 'high': bar.high, 'low': bar.low,
                'close': bar.close, 'volume': bar.volume
            } for bar in historical_data])
            
            df_with_indicators = indicators.calculate_all_indicators(df)
            latest = df_with_indicators.iloc[-1]
            current_price = latest['close']
            
            print(f"   💲 Current Price: ${current_price:.2f}")
            print(f"   📊 Volume: {latest['volume']:,.0f}")
            
            # Check if price is suitable for trading
            if current_price < 1.0:
                print(f"   ❌ Price too low (${current_price:.2f} < $1.00)")
                reasons_no_trades.append(f"{symbol}: Price too low for trading")
                continue
            
            # Check volume
            if latest['volume'] < 1000:
                print(f"   ❌ Volume too low ({latest['volume']:,.0f} < 1,000)")
                reasons_no_trades.append(f"{symbol}: Volume too low")
                continue
            
            # Try to generate signal
            mock_ohlcv = OHLCV(
                timestamp=datetime.now(),
                open=latest['open'], high=latest['high'], low=latest['low'],
                close=current_price, volume=latest['volume'],
                symbol=symbol, timeframe="1d"
            )
            mock_market_data = MarketData(
                symbol=symbol, price=current_price, volume=latest['volume'],
                timestamp=datetime.now(), data_type="trade", ohlcv=mock_ohlcv
            )
            
            signal = await signal_generator.generate_signal(symbol, mock_market_data, df_with_indicators)
            
            if signal:
                print(f"   🎯 Signal Generated: {signal.signal_type.value}")
                print(f"   📈 Confidence: {signal.confidence:.1%}")
                print(f"   💪 Strength: {signal.strength.value}")
                
                # Check why signal wasn't executed
                if signal.confidence < trading.pattern_threshold:
                    print(f"   ❌ Confidence too low ({signal.confidence:.1%} < {trading.pattern_threshold:.1%})")
                    reasons_no_trades.append(f"{symbol}: Signal confidence below threshold")
                else:
                    # Check position sizing
                    try:
                        position_size = risk_manager.calculate_position_size(signal, account_info)
                        if position_size <= 0:
                            print(f"   ❌ Position size too small ({position_size})")
                            reasons_no_trades.append(f"{symbol}: Position size calculation resulted in 0 shares")
                        else:
                            print(f"   ✅ Signal qualifies for trading! Position size: {position_size} shares")
                            print(f"   🤷 This signal should have triggered a trade...")
                    except Exception as e:
                        print(f"   ❌ Position sizing error: {e}")
                        reasons_no_trades.append(f"{symbol}: Position sizing error")
            else:
                print(f"   ⏸️  No signal generated")
                
                # Check why no signal
                if model.is_trained:
                    try:
                        features = model.prepare_features(df_with_indicators)
                        if not features.empty:
                            prediction = model.predict(features.iloc[-1:].values)
                            pred_confidence = max(prediction[0], 1 - prediction[0])
                            print(f"   🤖 AI Prediction: {prediction[0]:.3f} (confidence: {pred_confidence:.1%})")
                            
                            if pred_confidence < 0.6:
                                reasons_no_trades.append(f"{symbol}: AI prediction confidence too low")
                            else:
                                reasons_no_trades.append(f"{symbol}: Signal filters blocked the trade")
                    except Exception as e:
                        print(f"   ❌ Prediction error: {e}")
                        reasons_no_trades.append(f"{symbol}: ML prediction error")
                
                # Check technical indicators
                if 'rsi' in df_with_indicators.columns and not pd.isna(latest['rsi']):
                    rsi = latest['rsi']
                    print(f"   📊 RSI: {rsi:.1f}")
                    if 30 < rsi < 70:
                        print(f"   ℹ️  RSI in neutral zone (30-70)")
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
            reasons_no_trades.append(f"{symbol}: Analysis error - {e}")
    
    # Summary
    print(f"\n📋 SUMMARY: Why No Trades Yet?")
    print("=" * 60)
    
    if not reasons_no_trades:
        print("🤷 No obvious issues found. The bot should be ready to trade when conditions are right.")
        print("\n💡 Possible reasons:")
        print("   • Market conditions don't meet trading criteria yet")
        print("   • Signal cooldown period is active")
        print("   • Risk management is being very conservative")
        print("   • Waiting for stronger signals")
    else:
        print("🚨 Issues preventing trades:")
        for i, reason in enumerate(reasons_no_trades, 1):
            print(f"   {i}. {reason}")
    
    print(f"\n🔧 Suggestions to encourage trading:")
    print("   • Lower pattern_threshold in .env (currently {})".format(trading.pattern_threshold))
    print("   • Increase max_position_size if you want larger trades")
    print("   • Check that real-time data is flowing properly")
    print("   • Ensure your symbols are actively traded")
    print("   • Consider adding more volatile symbols")
    
    print(f"\n⏰ Market Status:")
    now = datetime.now()
    if 9 <= now.hour <= 16:  # Rough market hours
        print("   📈 Market likely open - good time for signals")
    else:
        print("   🌙 Market likely closed - signals may be limited")
    
    await provider.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(analyze_why_no_trades())
    except Exception as e:
        print(f"❌ Error: {e}") 