#!/usr/bin/env python3
"""
Simple script to check trading bot status and recent activity.
"""

import os
import json
import glob
from datetime import datetime

def check_bot_status():
    """Check the current status of the trading bot."""
    print("🤖 Trading Bot Status Check")
    print("=" * 50)
    
    # Check if bot is running
    import subprocess
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    bot_running = 'python3.11 src/main.py' in result.stdout
    
    print(f"Bot Running: {'✅ Yes' if bot_running else '❌ No'}")
    
    # Check latest log file
    log_files = glob.glob('logs/trading_bot_*.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"Latest Log: {os.path.basename(latest_log)}")
        
        # Read last few lines
        with open(latest_log, 'r') as f:
            lines = f.readlines()
            if lines:
                print(f"Last Activity: {lines[-1].strip()}")
                
                # Parse JSON log
                try:
                    last_event = json.loads(lines[-1])
                    event = last_event.get('event', '')
                    timestamp = last_event.get('timestamp', '')
                    print(f"Event: {event}")
                    print(f"Time: {timestamp}")
                except:
                    pass
    else:
        print("No log files found")
    
    # Check account status
    print("\n📊 Account Status:")
    try:
        import sys
        sys.path.insert(0, 'src')
        from data.hybrid_provider import HybridDataProvider
        import asyncio
        
        async def check_account():
            provider = HybridDataProvider()
            await provider.connect()
            account = await provider.get_account_info()
            await provider.disconnect()
            return account
        
        account = asyncio.run(check_account())
        if 'error' not in account:
            print(f"   Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
            print(f"   Cash: ${account.get('cash', 0):,.2f}")
            print(f"   Buying Power: ${account.get('buying_power', 0):,.2f}")
        else:
            print(f"   Error: {account.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Error checking account: {e}")
    
    # Check recent data
    print("\n📈 Recent Market Data:")
    try:
        import asyncio
        from datetime import datetime, timedelta
        
        async def check_data():
            provider = HybridDataProvider()
            await provider.connect()
            
            # Get latest AAPL data
            latest = await provider.get_latest_data("AAPL", "1d")
            await provider.disconnect()
            
            if latest:
                print(f"   AAPL Current Price: ${latest.close:.2f}")
                print(f"   Volume: {latest.volume:,.0f}")
                print(f"   Time: {latest.timestamp}")
            else:
                print("   No recent data available")
        
        asyncio.run(check_data())
    except Exception as e:
        print(f"   Error checking data: {e}")
    
    print("\n" + "=" * 50)
    if bot_running:
        print("🎉 Your trading bot is running successfully!")
        print("   It's processing real market data and ready to trade.")
    else:
        print("⚠️  Bot is not currently running.")
        print("   Start it with: python3.11 src/main.py --mode paper")

if __name__ == "__main__":
    check_bot_status() 