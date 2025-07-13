#!/usr/bin/env python3
"""
Simple backtesting script using yfinance for historical data.
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.technical_indicators import TechnicalIndicators


class SimpleBacktest:
    """Simple backtesting engine using yfinance data."""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.indicators = TechnicalIndicators()
        
    def get_data(self, symbol, start_date, end_date, interval='1h'):
        """Get historical data using yfinance."""
        print(f"📥 Downloading data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return None
            
        print(f"✅ Downloaded {len(data)} data points for {symbol}")
        return data
    
    def calculate_signals(self, data):
        """Calculate trading signals using technical indicators."""
        print("🔍 Calculating technical indicators...")
        
        # Calculate indicators
        data['sma_20'] = self.indicators.sma(data['Close'], 20)
        data['sma_50'] = self.indicators.sma(data['Close'], 50)
        data['rsi'] = self.indicators.rsi(data['Close'], 14)
        data['macd'], data['macd_signal'] = self.indicators.macd(data['Close'])
        
        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Simple strategy: Buy when SMA20 > SMA50 and RSI < 70, Sell when opposite
        buy_condition = (data['sma_20'] > data['sma_50']) & (data['rsi'] < 70)
        sell_condition = (data['sma_20'] < data['sma_50']) & (data['rsi'] > 30)
        
        signals.loc[buy_condition, 'signal'] = 1  # Buy
        signals.loc[sell_condition, 'signal'] = -1  # Sell
        
        return signals
    
    def run_backtest(self, symbol, start_date, end_date, interval='1h'):
        """Run the backtest."""
        print(f"🧪 Running backtest for {symbol}")
        print(f"   Period: {start_date} to {end_date}")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print()
        
        # Get data
        data = self.get_data(symbol, start_date, end_date, interval)
        if data is None:
            return None
        
        # Calculate signals
        signals = self.calculate_signals(data)
        
        # Run simulation
        print("💰 Running trading simulation...")
        portfolio = self.simulate_trading(data, signals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio, data)
        
        return {
            'data': data,
            'signals': signals,
            'portfolio': portfolio,
            'metrics': metrics
        }
    
    def simulate_trading(self, data, signals):
        """Simulate trading based on signals."""
        portfolio = pd.DataFrame(index=data.index)
        portfolio['cash'] = self.initial_capital
        portfolio['shares'] = 0
        portfolio['value'] = self.initial_capital
        portfolio['total_value'] = self.initial_capital
        
        position = 0
        entry_price = 0
        
        for i in range(1, len(data)):
            current_price = data['Close'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Update portfolio value
            portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio['cash'].iloc[i-1]
            portfolio.iloc[i, portfolio.columns.get_loc('shares')] = portfolio['shares'].iloc[i-1]
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy signal
                shares_to_buy = portfolio['cash'].iloc[i-1] * 0.95 / current_price  # Use 95% of cash
                cost = shares_to_buy * current_price * (1 + self.commission)
                
                if cost <= portfolio['cash'].iloc[i-1]:
                    portfolio.iloc[i, portfolio.columns.get_loc('shares')] = shares_to_buy
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio['cash'].iloc[i-1] - cost
                    position = 1
                    entry_price = current_price
                    print(f"   BUY: {shares_to_buy:.2f} shares at ${current_price:.2f}")
            
            elif signal == -1 and position == 1:  # Sell signal
                shares_to_sell = portfolio['shares'].iloc[i-1]
                revenue = shares_to_sell * current_price * (1 - self.commission)
                
                portfolio.iloc[i, portfolio.columns.get_loc('shares')] = 0
                portfolio.iloc[i, portfolio.columns.get_loc('cash')] = portfolio['cash'].iloc[i-1] + revenue
                position = 0
                
                pnl = revenue - (shares_to_sell * entry_price)
                print(f"   SELL: {shares_to_sell:.2f} shares at ${current_price:.2f} (PnL: ${pnl:.2f})")
            
            # Update total value
            portfolio.iloc[i, portfolio.columns.get_loc('value')] = (
                portfolio['shares'].iloc[i] * current_price
            )
            portfolio.iloc[i, portfolio.columns.get_loc('total_value')] = (
                portfolio['cash'].iloc[i] + portfolio['value'].iloc[i]
            )
        
        return portfolio
    
    def calculate_metrics(self, portfolio, data):
        """Calculate performance metrics."""
        total_return = (portfolio['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        daily_returns = portfolio['total_value'].pct_change().dropna()
        
        # Annualized return (assuming 252 trading days)
        annualized_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        
        # Sharpe ratio
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        trades = portfolio['shares'].diff().abs() > 0
        total_trades = trades.sum()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'final_value': portfolio['total_value'].iloc[-1]
        }
    
    def plot_results(self, results, filename=None):
        """Plot backtest results."""
        if results is None:
            print("❌ No results to plot")
            return
        
        data = results['data']
        signals = results['signals']
        portfolio = results['portfolio']
        metrics = results['metrics']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Signals', 'Portfolio Value', 'Technical Indicators'),
            vertical_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Price and signals
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add buy/sell signals
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        fig.add_trace(
            go.Scatter(x=buy_signals.index, y=data.loc[buy_signals.index, 'Close'],
                      mode='markers', name='Buy Signal', marker=dict(color='green', size=8)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sell_signals.index, y=data.loc[sell_signals.index, 'Close'],
                      mode='markers', name='Sell Signal', marker=dict(color='red', size=8)),
            row=1, col=1
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=portfolio.index, y=portfolio['total_value'], name='Portfolio Value',
                      line=dict(color='purple')),
            row=2, col=1
        )
        
        # Technical indicators
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', line=dict(color='orange')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50', line=dict(color='red')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Backtest Results - Total Return: {metrics["total_return"]:.2%}',
            height=800,
            showlegend=True
        )
        
        # Save plot
        if filename:
            fig.write_html(filename)
            print(f"📈 Plot saved to: {filename}")
        else:
            fig.show()


@click.command()
@click.option('--symbol', default='AAPL', help='Symbol to backtest')
@click.option('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date (YYYY-MM-DD)')
@click.option('--interval', default='1d', help='Data interval (1m, 5m, 15m, 1h, 1d)')
@click.option('--initial-capital', default=100000, help='Initial capital')
def main(symbol, start_date, end_date, interval, initial_capital):
    """Run a simple backtest."""
    
    # Create backtest engine
    backtest = SimpleBacktest(initial_capital=initial_capital)
    
    # Run backtest
    results = backtest.run_backtest(symbol, start_date, end_date, interval)
    
    if results is None:
        print("❌ Backtest failed")
        return
    
    # Display results
    metrics = results['metrics']
    print("\n📊 Backtest Results")
    print("=" * 50)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
    
    # Generate plot
    plot_file = f"backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    backtest.plot_results(results, plot_file)


if __name__ == "__main__":
    main() 