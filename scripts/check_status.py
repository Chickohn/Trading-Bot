#!/usr/bin/env python3
"""
Script to check the current status of the trading bot.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_bot_status():
    """Check the current status of the trading bot."""
    print("🤖 Trading Bot Status Check")
    print("=" * 50)
    
    # Check if bot is running
    print("\n📊 Process Status:")
    try:
        import psutil
        bot_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower() and 'main.py' in ' '.join(proc.info['cmdline'] or []):
                    bot_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if bot_processes:
            print("✅ Trading bot is running!")
            for proc in bot_processes:
                print(f"   PID: {proc['pid']}")
        else:
            print("❌ Trading bot is not running")
    except ImportError:
        print("⚠️  psutil not available - can't check process status")
    
    # Check recent logs
    print("\n📝 Recent Logs:")
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            # Get the most recent log file
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"📄 Latest log file: {latest_log.name}")
            
            # Show last 10 lines
            try:
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print("\nLast 10 log entries:")
                        for line in lines[-10:]:
                            try:
                                log_entry = json.loads(line.strip())
                                timestamp = log_entry.get('timestamp', '')
                                event = log_entry.get('event', '')
                                level = log_entry.get('level', '')
                                print(f"   [{timestamp}] {level.upper()}: {event}")
                            except json.JSONDecodeError:
                                print(f"   {line.strip()}")
                    else:
                        print("   No log entries found")
            except Exception as e:
                print(f"   Error reading log file: {e}")
        else:
            print("   No log files found")
    else:
        print("   Logs directory not found")
    
    # Check configuration
    print("\n⚙️  Configuration:")
    try:
        from utils.config import config, trading, alpaca
        print(f"   Environment: {config.environment}")
        print(f"   Trading symbols: {trading.trading_symbols}")
        print(f"   Default timeframe: {trading.default_timeframe}")
        print(f"   Alpaca API configured: {'Yes' if alpaca.api_key else 'No'}")
    except Exception as e:
        print(f"   Error loading config: {e}")
    
    # Check data directory
    print("\n📁 Data Files:")
    data_dir = Path("data")
    if data_dir.exists():
        data_files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.json"))
        if data_files:
            print(f"   Found {len(data_files)} data files")
            for file in data_files[:5]:  # Show first 5
                print(f"   - {file.name}")
            if len(data_files) > 5:
                print(f"   ... and {len(data_files) - 5} more")
        else:
            print("   No data files found")
    else:
        print("   Data directory not found")
    
    # Check models
    print("\n🧠 Models:")
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.pkl")) + list(models_dir.rglob("*.joblib"))
        if model_files:
            print(f"   Found {len(model_files)} model files")
            for file in model_files:
                print(f"   - {file.name}")
        else:
            print("   No trained models found")
    else:
        print("   Models directory not found")

if __name__ == "__main__":
    check_bot_status() 