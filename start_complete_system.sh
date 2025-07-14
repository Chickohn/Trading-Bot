#!/bin/bash

# Complete Trading Bot System Startup Script
# Starts: Trading Bot + Dashboard + Bridge

echo "🚀 Starting Complete Trading Bot System..."
echo "🤖 This will start:"
echo "   - Trading Bot (executes trades)"
echo "   - Dashboard (visualization)"
echo "   - Bridge (connects them)"
echo ""

# Activate virtual environment
source venv/bin/activate

# Function to check if a process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to start a component
start_component() {
    local name=$1
    local command=$2
    local log_file=$3
    
    if is_running "$name"; then
        echo "✅ $name is already running"
    else
        echo "🚀 Starting $name..."
        nohup python $command > $log_file 2>&1 &
        sleep 2
        if is_running "$name"; then
            echo "✅ $name started successfully"
        else
            echo "❌ Failed to start $name"
        fi
    fi
}

# Create logs directory
mkdir -p logs

# Start components
echo "📊 Starting components..."

# 1. Start the bridge (connects bot to dashboard)
start_component "bot_dashboard_bridge" "bot_dashboard_bridge.py" "logs/bridge.log"

# 2. Start the trading bot
start_component "src/main.py" "src/main.py --mode paper" "logs/bot.log"

# 3. Start the dashboard
start_component "comprehensive_dashboard" "comprehensive_dashboard.py" "logs/dashboard.log"

# Wait a moment for everything to start
sleep 5

# Check status
echo ""
echo "📋 System Status:"
echo "=================="

if is_running "bot_dashboard_bridge"; then
    echo "✅ Bridge: Running (monitoring bot activity)"
else
    echo "❌ Bridge: Not running"
fi

if is_running "src/main.py"; then
    echo "✅ Trading Bot: Running (executing trades)"
else
    echo "❌ Trading Bot: Not running"
fi

if is_running "comprehensive_dashboard"; then
    echo "✅ Dashboard: Running (http://localhost:8050)"
else
    echo "❌ Dashboard: Not running"
fi

echo ""
echo "🌐 Access Points:"
echo "   - Dashboard: http://localhost:8050"
echo "   - Bot Logs: tail -f logs/bot.log"
echo "   - Bridge Logs: tail -f logs/bridge.log"
echo ""
echo "🛑 To stop everything: ./stop_system.sh"
echo ""

# Show recent activity
echo "📊 Recent Activity:"
echo "=================="
tail -5 logs/bot.log 2>/dev/null || echo "No bot logs yet"
echo ""
tail -5 logs/bridge.log 2>/dev/null || echo "No bridge logs yet" 