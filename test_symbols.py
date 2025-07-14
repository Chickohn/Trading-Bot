#!/usr/bin/env python3
"""
Test script to verify symbol loading and data fetching.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import trading
from data.hybrid_provider import HybridDataProvider

async def test_symbols():
    """Test symbol loading and data fetching."""
    print("🧪 Testing Symbol Loading and Data Fetching")
    print("=" * 50)
    
    # Test 1: Check if symbols are loaded
    print(f"\n📊 Trading Symbols: {trading.trading_symbols}")
    print(f"📊 Number of symbols: {len(trading.trading_symbols)}")
    
    # Test 2: Test data provider connection
    print("\n🔗 Testing Data Provider Connection...")
    provider = HybridDataProvider()
    
    try:
        connected = await provider.connect()
        if connected:
            print("✅ Successfully connected to data provider")
        else:
            print("❌ Failed to connect to data provider")
            return
    except Exception as e:
        print(f"❌ Error connecting to data provider: {e}")
        return
    
    # Test 3: Test data fetching for first few symbols
    print("\n📈 Testing Data Fetching...")
    test_symbols = trading.trading_symbols[:5]  # Test first 5 symbols
    
    for symbol in test_symbols:
        try:
            print(f"\n🔍 Testing {symbol}...")
            
            # Get latest data
            latest_data = await provider.get_latest_data(symbol, "1d")
            if latest_data:
                print(f"  ✅ Latest price: ${latest_data.close:.2f}")
                print(f"  ✅ Volume: {latest_data.volume:,.0f}")
                print(f"  ✅ Timestamp: {latest_data.timestamp}")
            else:
                print(f"  ❌ No data available for {symbol}")
                
        except Exception as e:
            print(f"  ❌ Error fetching data for {symbol}: {e}")
    
    # Test 4: Test historical data
    print("\n📊 Testing Historical Data...")
    symbol = trading.trading_symbols[0]  # Test first symbol
    
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days
        
        historical_data = await provider.get_historical_data(
            symbol, "1d", start_date, end_date
        )
        
        if historical_data:
            print(f"  ✅ Retrieved {len(historical_data)} historical bars for {symbol}")
            print(f"  ✅ Date range: {historical_data[0].timestamp} to {historical_data[-1].timestamp}")
        else:
            print(f"  ❌ No historical data available for {symbol}")
            
    except Exception as e:
        print(f"  ❌ Error fetching historical data: {e}")
    
    # Cleanup
    await provider.disconnect()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_symbols()) 