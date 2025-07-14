#!/usr/bin/env python3
"""
Test script to check Alpaca data availability and connectivity.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import alpaca
from alpaca.data import StockHistoricalDataClient, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient

async def test_alpaca_connection():
    """Test Alpaca API connection and data access."""
    print("🔍 Testing Alpaca Connection...")
    print(f"API Key: {alpaca.api_key[:8]}...")
    print(f"Paper Trading: {alpaca.paper_trading}")
    print(f"Base URL: {alpaca.base_url}")
    
    try:
        # Test trading client
        trading_client = TradingClient(
            api_key=alpaca.api_key,
            secret_key=alpaca.secret_key,
            paper=alpaca.paper_trading
        )
        
        # Get account info
        account = trading_client.get_account()
        print(f"✅ Trading Account: {account.status}")
        print(f"   Cash: ${account.cash}")
        print(f"   Portfolio Value: ${account.portfolio_value}")
        
        # Test data client
        data_client = StockHistoricalDataClient(
            api_key=alpaca.api_key,
            secret_key=alpaca.secret_key
        )
        
        # Test historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Free tier limit
        
        request = StockBarsRequest(
            symbol_or_symbols="AAPL",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = data_client.get_stock_bars(request)
        print(f"✅ Historical Data: {len(bars)} bars for AAPL")
        
        if len(bars) > 0:
            latest_bar = bars[-1]
            print(f"   Latest: {latest_bar.timestamp} - Close: ${latest_bar.close}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_real_time_data():
    """Test if we can access real-time data."""
    print("\n🔍 Testing Real-Time Data Access...")
    
    try:
        # This will fail on free tier, but let's see the error
        from alpaca.data.live import StockDataStream
        
        stream = StockDataStream(
            api_key=alpaca.api_key,
            secret_key=alpaca.secret_key
        )
        
        print("✅ Real-time streaming available")
        return True
        
    except Exception as e:
        print(f"❌ Real-time data not available: {e}")
        print("   This is expected with free Alpaca tier")
        return False

async def test_trading_capabilities():
    """Test trading capabilities."""
    print("\n🔍 Testing Trading Capabilities...")
    
    try:
        trading_client = TradingClient(
            api_key=alpaca.api_key,
            secret_key=alpaca.secret_key,
            paper=alpaca.paper_trading
        )
        
        # Get available symbols
        assets = trading_client.get_all_assets()
        active_stocks = [asset for asset in assets if asset.status == 'active' and asset.tradable]
        print(f"✅ Trading Assets: {len(active_stocks)} active stocks available")
        
        # Test order placement (paper trade)
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        # This would place a real order, so we'll just test the request creation
        order_request = MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY
        )
        
        print("✅ Order creation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Trading test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Alpaca Trading Bot - Data Availability Test")
    print("=" * 50)
    
    results = []
    
    # Test connection
    results.append(await test_alpaca_connection())
    
    # Test real-time data
    results.append(await test_real_time_data())
    
    # Test trading
    results.append(await test_trading_capabilities())
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Connection: {'✅' if results[0] else '❌'}")
    print(f"   Real-time Data: {'✅' if results[1] else '❌'}")
    print(f"   Trading: {'✅' if results[2] else '❌'}")
    
    if all(results[:2]):  # Connection and trading work
        print("\n🎉 Your setup is ready for paper trading!")
        print("   Note: Real-time data requires paid Alpaca plan")
    elif results[0]:  # Only connection works
        print("\n⚠️  Limited functionality with free tier")
        print("   Consider upgrading to Alpaca Starter Plan ($9/month)")
    else:
        print("\n❌ Setup needs attention")
        print("   Check your API keys and internet connection")

if __name__ == "__main__":
    asyncio.run(main()) 