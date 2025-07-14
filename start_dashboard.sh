#!/bin/bash

# Comprehensive Trading Bot Dashboard Startup Script

echo "🚀 Starting Comprehensive Trading Bot Dashboard..."
echo "📊 This will show all 25+ symbols with real-time analysis"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if dashboard dependencies are installed
python -c "import dash, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing dashboard dependencies..."
    pip install dash dash-bootstrap-components plotly
fi

# Start the comprehensive dashboard
echo "🌐 Dashboard will be available at: http://localhost:8050"
echo "📱 Open your browser and navigate to the URL above"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

python comprehensive_dashboard.py 