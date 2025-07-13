#!/usr/bin/env python3
"""
Backtesting script for trading strategies.
"""

import os
import sys
import asyncio
import click
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.backtest_engine import BacktestEngine
from utils.config import trading, risk
from data.alpaca_provider import AlpacaDataProvider


@click.command()
@click.option('--symbols', default='AAPL,TSLA', help='Comma-separated list of symbols to backtest')
@click.option('--start-date', default='2024-01-01', help='Start date for backtest (YYYY-MM-DD)')
@click.option('--end-date', default='2024-12-31', help='End date for backtest (YYYY-MM-DD)')
@click.option('--timeframe', default='1h', help='Timeframe for data (1m, 5m, 15m, 1h, 4h, 1d)')
@click.option('--initial-capital', default=100000, help='Initial capital for backtest')
@click.option('--commission', default=0.001, help='Commission rate (0.001 = 0.1%)')
@click.option('--output-file', help='Output file for results (CSV)')
def run_backtest(symbols, start_date, end_date, timeframe, initial_capital, commission, output_file):
    """Run a backtest on historical data."""
    
    print("🧪 Starting Backtest")
    print("=" * 50)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Commission: {commission:.3f}")
    print()
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        print(f"❌ Error parsing dates: {e}")
        return
    
    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        symbols=symbol_list,
        timeframe=timeframe
    )
    
    # Run backtest
    try:
        results = asyncio.run(engine.run_backtest(start_dt, end_dt))
        
        # Display results
        print("\n📊 Backtest Results")
        print("=" * 50)
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Annualized Return: {results['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.3f}")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        
        # Save results if output file specified
        if output_file:
            engine.save_results(results, output_file)
            print(f"\n💾 Results saved to: {output_file}")
        
        # Generate plots
        plot_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        engine.plot_results(results, plot_file)
        print(f"📈 Interactive plot saved to: {plot_file}")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


@click.command()
@click.option('--symbol', default='AAPL', help='Symbol to analyze')
@click.option('--days', default=30, help='Number of days to analyze')
def analyze_data(symbol, days):
    """Analyze historical data for a symbol."""
    
    print(f"📈 Data Analysis for {symbol}")
    print("=" * 50)
    
    async def analyze():
        # Create data provider
        provider = AlpacaDataProvider()
        await provider.connect()
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = await provider.get_historical_data(
            symbol, "1h", start_date, end_date
        )
        
        if not data:
            print(f"❌ No data found for {symbol}")
            return
        
        print(f"📊 Data Summary:")
        print(f"   Period: {start_date.date()} to {end_date.date()}")
        print(f"   Data points: {len(data)}")
        print(f"   First price: ${data[0].close:.2f}")
        print(f"   Last price: ${data[-1].close:.2f}")
        print(f"   Price change: {((data[-1].close - data[0].close) / data[0].close * 100):.2f}%")
        
        # Calculate some basic stats
        prices = [d.close for d in data]
        volumes = [d.volume for d in data]
        
        print(f"\n📈 Price Statistics:")
        print(f"   Min: ${min(prices):.2f}")
        print(f"   Max: ${max(prices):.2f}")
        print(f"   Average: ${sum(prices)/len(prices):.2f}")
        print(f"   Volatility: {((max(prices) - min(prices)) / min(prices) * 100):.2f}%")
        
        print(f"\n📊 Volume Statistics:")
        print(f"   Total volume: {sum(volumes):,.0f}")
        print(f"   Average volume: {sum(volumes)/len(volumes):,.0f}")
        
        await provider.disconnect()
    
    asyncio.run(analyze())


@click.group()
def cli():
    """Trading Bot Backtesting Tools"""
    pass


cli.add_command(run_backtest)
cli.add_command(analyze_data)

if __name__ == "__main__":
    cli() 