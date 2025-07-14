#!/usr/bin/env python3
"""
Real-time trading bot dashboard to visualize analysis and signals.
"""

import sys
import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.hybrid_provider import HybridDataProvider
from features.technical_indicators import TechnicalIndicators
from models.random_forest_model import RandomForestModel
from models.base_model import ModelConfig
from signals.signal_generator import SignalGenerator
from risk.risk_manager import RiskManager

class TradingDashboard:
    """Real-time trading dashboard."""
    
    def __init__(self):
        self.symbols = ["AAPL", "TSLA", "MSFT", "SPX.L"]
        self.provider = HybridDataProvider()
        self.indicators = TechnicalIndicators()
        self.data_history = {}
        self.signals_history = []
        
        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Trading Bot Real-Time Dashboard', color='white', fontsize=16)
        
    async def initialize(self):
        """Initialize the dashboard."""
        await self.provider.connect()
        print("📊 Dashboard initialized - fetching initial data...")
        
        # Get initial data for all symbols
        for symbol in self.symbols:
            await self.fetch_data(symbol)
    
    async def fetch_data(self, symbol: str):
        """Fetch latest data for a symbol."""
        try:
            # Get recent historical data - request more days to ensure sufficient data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # Get full year for better indicators
            
            historical_data = await self.provider.get_historical_data(
                symbol, "1d", start_date, end_date
            )
            
            if historical_data:
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                } for bar in historical_data])
                
                df.set_index('timestamp', inplace=True)
                df = df.sort_index()
                
                # Calculate technical indicators
                df_with_indicators = self.indicators.calculate_all_indicators(df)
                
                self.data_history[symbol] = df_with_indicators
                
                print(f"✅ Fetched {len(df)} bars for {symbol}")
                return df_with_indicators
                
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Analyze a symbol and return trading signals/analysis."""
        try:
            if df.empty or len(df) < 20:
                return {"error": "Insufficient data"}
            
            latest = df.iloc[-1]
            
            # Technical Analysis Summary
            analysis = {
                "symbol": symbol,
                "current_price": latest['close'],
                "volume": latest['volume'],
                "timestamp": latest.name,
                "indicators": {},
                "signals": {},
                "trend": "neutral"
            }
            
            # Key indicators
            if 'sma_20' in df.columns and not pd.isna(latest['sma_20']):
                analysis["indicators"]["SMA20"] = latest['sma_20']
                analysis["signals"]["price_vs_sma20"] = "bullish" if latest['close'] > latest['sma_20'] else "bearish"
            
            if 'rsi' in df.columns and not pd.isna(latest['rsi']):
                analysis["indicators"]["RSI"] = latest['rsi']
                if latest['rsi'] > 70:
                    analysis["signals"]["rsi_signal"] = "overbought"
                elif latest['rsi'] < 30:
                    analysis["signals"]["rsi_signal"] = "oversold"
                else:
                    analysis["signals"]["rsi_signal"] = "neutral"
            
            if 'macd' in df.columns and not pd.isna(latest['macd']):
                analysis["indicators"]["MACD"] = latest['macd']
                if 'macd_signal' in df.columns and not pd.isna(latest['macd_signal']):
                    analysis["signals"]["macd_signal"] = "bullish" if latest['macd'] > latest['macd_signal'] else "bearish"
            
            # Overall trend analysis
            if 'sma_5' in df.columns and 'sma_20' in df.columns:
                if not pd.isna(latest['sma_5']) and not pd.isna(latest['sma_20']):
                    if latest['sma_5'] > latest['sma_20']:
                        analysis["trend"] = "bullish"
                    else:
                        analysis["trend"] = "bearish"
            
            # Price change
            if len(df) > 1:
                prev_close = df.iloc[-2]['close']
                price_change = ((latest['close'] - prev_close) / prev_close) * 100
                analysis["price_change_pct"] = price_change
            
            return analysis
            
        except Exception as e:
            return {"error": f"Analysis error: {e}"}
    
    def update_dashboard(self):
        """Update the dashboard plots."""
        try:
            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()
            
            # Plot 1: Price and Moving Averages
            ax1 = self.axes[0, 0]
            ax1.set_title('Price & Moving Averages', color='white')
            
            for i, symbol in enumerate(self.symbols[:3]):
                if symbol in self.data_history:
                    df = self.data_history[symbol]
                    if not df.empty:
                        # Plot last 20 days
                        recent_df = df.tail(20)
                        color = ['cyan', 'yellow', 'magenta'][i]
                        
                        ax1.plot(recent_df.index, recent_df['close'], 
                                label=f'{symbol} Price', color=color, linewidth=2)
                        
                        if 'sma_20' in recent_df.columns:
                            ax1.plot(recent_df.index, recent_df['sma_20'], 
                                    '--', alpha=0.7, color=color, label=f'{symbol} SMA20')
            
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: RSI
            ax2 = self.axes[0, 1]
            ax2.set_title('RSI Oscillator', color='white')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
            ax2.axhline(y=50, color='white', linestyle='-', alpha=0.5)
            
            for i, symbol in enumerate(self.symbols[:3]):
                if symbol in self.data_history:
                    df = self.data_history[symbol]
                    if not df.empty and 'rsi' in df.columns:
                        recent_df = df.tail(20)
                        color = ['cyan', 'yellow', 'magenta'][i]
                        ax2.plot(recent_df.index, recent_df['rsi'], 
                                label=f'{symbol} RSI', color=color, linewidth=2)
            
            ax2.set_ylim(0, 100)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Volume
            ax3 = self.axes[1, 0]
            ax3.set_title('Volume Analysis', color='white')
            
            for i, symbol in enumerate(self.symbols[:3]):
                if symbol in self.data_history:
                    df = self.data_history[symbol]
                    if not df.empty:
                        recent_df = df.tail(20)
                        color = ['cyan', 'yellow', 'magenta'][i]
                        ax3.bar(recent_df.index, recent_df['volume'], 
                               alpha=0.6, color=color, label=f'{symbol} Volume')
            
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Analysis Summary
            ax4 = self.axes[1, 1]
            ax4.set_title('AI Analysis Summary', color='white')
            ax4.axis('off')
            
            summary_text = []
            for symbol in self.symbols:
                if symbol in self.data_history:
                    analysis = self.analyze_symbol(symbol, self.data_history[symbol])
                    if "error" not in analysis:
                        trend_color = {'bullish': '🟢', 'bearish': '🔴', 'neutral': '🟡'}
                        trend_emoji = trend_color.get(analysis['trend'], '🟡')
                        
                        price_change = analysis.get('price_change_pct', 0)
                        change_str = f"{price_change:+.2f}%" if price_change else "N/A"
                        
                        summary_text.append(f"{symbol}: ${analysis['current_price']:.2f} ({change_str})")
                        summary_text.append(f"  Trend: {trend_emoji} {analysis['trend'].upper()}")
                        
                        if 'RSI' in analysis['indicators']:
                            rsi_val = analysis['indicators']['RSI']
                            summary_text.append(f"  RSI: {rsi_val:.1f}")
                        
                        if 'rsi_signal' in analysis['signals']:
                            signal = analysis['signals']['rsi_signal']
                            summary_text.append(f"  Signal: {signal.upper()}")
                        
                        summary_text.append("")  # Empty line
            
            # Display summary
            ax4.text(0.05, 0.95, '\n'.join(summary_text), 
                    transform=ax4.transAxes, fontsize=10, 
                    verticalalignment='top', color='white',
                    fontfamily='monospace')
            
            plt.tight_layout()
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    async def run_dashboard(self):
        """Run the real-time dashboard."""
        await self.initialize()
        
        print("🚀 Starting real-time dashboard...")
        print("📊 Dashboard will update every 30 seconds")
        print("❌ Close the window to stop")
        
        def animate(frame):
            """Animation function for matplotlib."""
            # This runs in sync, so we can't use async here
            # We'll update manually in the main loop
            pass
        
        # Set up animation (though we'll update manually)
        ani = animation.FuncAnimation(self.fig, animate, interval=30000)
        
        # Main update loop
        try:
            while True:
                # Update data
                for symbol in self.symbols:
                    await self.fetch_data(symbol)
                
                # Update plots
                self.update_dashboard()
                plt.draw()
                plt.pause(0.1)
                
                # Wait 30 seconds
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
        finally:
            await self.provider.disconnect()
            plt.close()

async def main():
    """Run the dashboard."""
    print("🤖 Trading Bot Real-Time Dashboard")
    print("=" * 50)
    
    dashboard = TradingDashboard()
    await dashboard.run_dashboard()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}") 