#!/usr/bin/env python3
"""
Bot Brain Viewer - See what your trading bot is thinking in real-time.
"""

import sys
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.hybrid_provider import HybridDataProvider
from features.technical_indicators import TechnicalIndicators
from models.random_forest_model import RandomForestModel
from models.base_model import ModelConfig
from signals.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager
from data.base import MarketData, OHLCV

class BotBrainViewer:
    """Shows what the trading bot is analyzing and thinking."""
    
    def __init__(self):
        self.symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "SPX.L"]
        self.provider = HybridDataProvider()
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager()
        
        # Initialize ML model
        config = ModelConfig(
            model_name="brain_viewer",
            lookback_period=20,
            prediction_threshold=0.7,
            retrain_frequency="1d"
        )
        self.model = RandomForestModel(config)
        self.signal_generator = None
        
    async def initialize(self):
        """Initialize the brain viewer."""
        print("🧠 Initializing Bot Brain Viewer...")
        await self.provider.connect()
        
        # Train the model with recent data
        print("🎓 Training AI model...")
        await self.train_model()
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator([self.model])
        
        print("✅ Brain viewer ready!")
    
    async def train_model(self):
        """Train the ML model with recent data."""
        try:
            # Get training data from AAPL (our primary symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 year of data
            
            historical_data = await self.provider.get_historical_data(
                "AAPL", "1d", start_date, end_date
            )
            
            if len(historical_data) < 100:
                print("⚠️  Limited training data available")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in historical_data])
            
            # Add target (next day price direction)
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Prepare features and train
            X = self.model.prepare_features(df)
            y = self.model.prepare_target(df)
            
            # Align data
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) > 50:
                metrics = self.model.train(X, y)
                print(f"🎯 Model trained with {metrics.get('accuracy', 0):.1%} accuracy")
            
        except Exception as e:
            print(f"❌ Training error: {e}")
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    async def analyze_symbol_deep(self, symbol: str) -> Dict:
        """Deep analysis of a symbol."""
        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            
            historical_data = await self.provider.get_historical_data(
                symbol, "1d", start_date, end_date
            )
            
            if not historical_data or len(historical_data) < 20:
                return {"error": "Insufficient data"}
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in historical_data])
            
            # Calculate technical indicators
            df_with_indicators = self.indicators.calculate_all_indicators(df)
            
            latest = df_with_indicators.iloc[-1]
            prev = df_with_indicators.iloc[-2]
            
            # Get latest market data
            latest_data = await self.provider.get_latest_data(symbol, "1d")
            current_price = latest_data.close if latest_data else latest['close']
            
            # AI Prediction
            ai_prediction = None
            ai_confidence = 0
            try:
                if self.model.is_trained:
                    features = self.model.prepare_features(df_with_indicators)
                    if not features.empty:
                        prediction = self.model.predict(features.iloc[-1:].values)
                        ai_prediction = "BUY" if prediction[0] > 0.5 else "SELL"
                        ai_confidence = max(prediction[0], 1 - prediction[0])
            except Exception as e:
                ai_prediction = "ERROR"
            
            # Technical Analysis
            analysis = {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": ((current_price - prev['close']) / prev['close']) * 100,
                "volume": latest['volume'],
                "volume_change": ((latest['volume'] - prev['volume']) / prev['volume']) * 100,
                
                # Technical Indicators
                "indicators": {},
                "signals": [],
                "trend_strength": 0,
                "ai_prediction": ai_prediction,
                "ai_confidence": ai_confidence
            }
            
            # Key Technical Indicators
            if 'sma_20' in df_with_indicators.columns and not pd.isna(latest['sma_20']):
                analysis["indicators"]["SMA20"] = latest['sma_20']
                if current_price > latest['sma_20']:
                    analysis["signals"].append("🟢 Price above SMA20 (Bullish)")
                else:
                    analysis["signals"].append("🔴 Price below SMA20 (Bearish)")
            
            if 'rsi' in df_with_indicators.columns and not pd.isna(latest['rsi']):
                analysis["indicators"]["RSI"] = latest['rsi']
                if latest['rsi'] > 70:
                    analysis["signals"].append("🔴 RSI Overbought (>70)")
                elif latest['rsi'] < 30:
                    analysis["signals"].append("🟢 RSI Oversold (<30)")
                else:
                    analysis["signals"].append(f"🟡 RSI Neutral ({latest['rsi']:.1f})")
            
            if 'macd' in df_with_indicators.columns and not pd.isna(latest['macd']):
                analysis["indicators"]["MACD"] = latest['macd']
                if 'macd_signal' in df_with_indicators.columns and not pd.isna(latest['macd_signal']):
                    if latest['macd'] > latest['macd_signal']:
                        analysis["signals"].append("🟢 MACD Bullish Crossover")
                    else:
                        analysis["signals"].append("🔴 MACD Bearish Crossover")
            
            # Volume Analysis
            if analysis["volume_change"] > 20:
                analysis["signals"].append(f"📈 High Volume (+{analysis['volume_change']:.1f}%)")
            elif analysis["volume_change"] < -20:
                analysis["signals"].append(f"📉 Low Volume ({analysis['volume_change']:.1f}%)")
            
            # Trend Strength Calculation
            trend_signals = [s for s in analysis["signals"] if "🟢" in s]
            bearish_signals = [s for s in analysis["signals"] if "🔴" in s]
            analysis["trend_strength"] = len(trend_signals) - len(bearish_signals)
            
            # Generate trading signal
            try:
                if self.signal_generator and latest_data:
                    mock_ohlcv = OHLCV(
                        timestamp=datetime.now(),
                        open=latest['open'],
                        high=latest['high'],
                        low=latest['low'],
                        close=current_price,
                        volume=latest['volume'],
                        symbol=symbol,
                        timeframe="1d"
                    )
                    mock_market_data = MarketData(
                        symbol=symbol,
                        price=current_price,
                        volume=latest['volume'],
                        timestamp=datetime.now(),
                        data_type="trade",
                        ohlcv=mock_ohlcv
                    )
                    
                    signal = await self.signal_generator.generate_signal(
                        symbol, mock_market_data, df_with_indicators
                    )
                    
                    if signal:
                        analysis["bot_signal"] = {
                            "action": signal.signal_type.value,
                            "confidence": signal.confidence,
                            "strength": signal.strength.value
                        }
                    else:
                        analysis["bot_signal"] = {"action": "HOLD", "reason": "No strong signal"}
            except Exception as e:
                analysis["bot_signal"] = {"action": "ERROR", "error": str(e)}
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    def display_analysis(self, analyses: List[Dict]):
        """Display the analysis in a nice format."""
        self.clear_screen()
        
        print("🧠 BOT BRAIN VIEWER - Real-Time Analysis")
        print("=" * 80)
        print(f"⏰ Updated: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)
        
        for analysis in analyses:
            if "error" in analysis:
                print(f"❌ {analysis.get('symbol', 'Unknown')}: {analysis['error']}")
                continue
            
            symbol = analysis["symbol"]
            price = analysis["current_price"]
            change = analysis["price_change"]
            
            # Header
            change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"\n🔍 {symbol}: ${price:.2f} {change_emoji} {change:+.2f}%")
            
            # AI Prediction
            if analysis["ai_prediction"]:
                confidence_bar = "█" * int(analysis["ai_confidence"] * 10)
                print(f"🤖 AI Prediction: {analysis['ai_prediction']} "
                      f"(Confidence: {analysis['ai_confidence']:.1%} {confidence_bar})")
            
            # Bot Signal
            if "bot_signal" in analysis:
                signal = analysis["bot_signal"]
                action = signal.get("action", "UNKNOWN")
                if action == "BUY":
                    print(f"🚀 BOT SIGNAL: {action} (Confidence: {signal.get('confidence', 0):.1%})")
                elif action == "SELL":
                    print(f"🔥 BOT SIGNAL: {action} (Confidence: {signal.get('confidence', 0):.1%})")
                else:
                    print(f"⏸️  BOT SIGNAL: {action}")
            
            # Technical Signals
            print("📊 Technical Analysis:")
            for signal in analysis["signals"][:4]:  # Show top 4 signals
                print(f"   {signal}")
            
            # Trend Strength
            strength = analysis["trend_strength"]
            if strength > 1:
                print(f"📈 Overall Trend: BULLISH (Strength: +{strength})")
            elif strength < -1:
                print(f"📉 Overall Trend: BEARISH (Strength: {strength})")
            else:
                print(f"↔️  Overall Trend: NEUTRAL (Strength: {strength})")
            
            print("-" * 60)
        
        print("\n💡 What the bot is thinking:")
        print("   • Analyzing price patterns and technical indicators")
        print("   • Using AI to predict next price movement")
        print("   • Evaluating risk and position sizing")
        print("   • Looking for high-confidence trading opportunities")
        print("\n🔄 Updates every 30 seconds... (Ctrl+C to stop)")
    
    async def run(self):
        """Run the brain viewer."""
        await self.initialize()
        
        try:
            while True:
                # Analyze all symbols
                analyses = []
                for symbol in self.symbols:
                    print(f"🔍 Analyzing {symbol}...")
                    analysis = await self.analyze_symbol_deep(symbol)
                    analyses.append(analysis)
                
                # Display results
                self.display_analysis(analyses)
                
                # Wait 30 seconds
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n👋 Bot Brain Viewer stopped")
        finally:
            await self.provider.disconnect()

async def main():
    """Run the bot brain viewer."""
    viewer = BotBrainViewer()
    await viewer.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}") 