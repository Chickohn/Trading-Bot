"""
Logging and monitoring system for the trading bot.
"""

import logging
import structlog
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import redis as redis_client
import asyncio

from utils.config import config, monitoring, redis as redis_config

# Prometheus metrics
SIGNALS_GENERATED = Counter('trading_signals_generated_total', 'Total signals generated', ['symbol', 'signal_type'])
TRADES_EXECUTED = Counter('trading_trades_executed_total', 'Total trades executed', ['symbol', 'side'])
POSITIONS_OPEN = Gauge('trading_positions_open', 'Number of open positions')
PORTFOLIO_VALUE = Gauge('trading_portfolio_value', 'Current portfolio value')
DAILY_PNL = Gauge('trading_daily_pnl', 'Daily profit/loss')
SIGNAL_LATENCY = Histogram('trading_signal_latency_seconds', 'Signal generation latency')
MODEL_PREDICTION_TIME = Histogram('trading_model_prediction_seconds', 'Model prediction time')


def setup_logging():
    """Setup structured logging with proper configuration."""
    import os
    from datetime import datetime
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/trading_bot_{timestamp}.log"
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to both file and stdout
    logging.basicConfig(
        level=getattr(logging, monitoring.log_level.upper()),
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log startup message
    logger = structlog.get_logger(__name__)
    logger.info(f"Logging initialized - file: {log_filename}")


class TradingLogger:
    """Enhanced logger for trading-specific events."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.redis_client = None
        self._setup_redis()
    
    def _setup_redis(self):
        """Setup Redis connection for caching and real-time data."""
        try:
            self.redis_client = redis_client.from_url(redis_config.url)
            self.redis_client.ping()
            self.logger.info("Connected to Redis")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log a trading signal."""
        self.logger.info(
            "Trading signal generated",
            symbol=signal_data.get('symbol'),
            signal_type=signal_data.get('signal_type'),
            confidence=signal_data.get('confidence'),
            strength=signal_data.get('strength'),
            price=signal_data.get('price'),
            models_used=signal_data.get('models_used', [])
        )
        
        # Update Prometheus metrics
        SIGNALS_GENERATED.labels(
            symbol=signal_data.get('symbol'),
            signal_type=signal_data.get('signal_type')
        ).inc()
        
        # Cache signal in Redis
        if self.redis_client:
            key = f"signal:{signal_data.get('symbol')}:{datetime.now().isoformat()}"
            self.redis_client.setex(key, 3600, json.dumps(signal_data))  # 1 hour TTL
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade execution."""
        self.logger.info(
            "Trade executed",
            symbol=trade_data.get('symbol'),
            side=trade_data.get('side'),
            quantity=trade_data.get('quantity'),
            price=trade_data.get('price'),
            order_id=trade_data.get('order_id'),
            execution_time=trade_data.get('execution_time')
        )
        
        # Update Prometheus metrics
        TRADES_EXECUTED.labels(
            symbol=trade_data.get('symbol'),
            side=trade_data.get('side')
        ).inc()
        
        # Cache trade in Redis
        if self.redis_client:
            key = f"trade:{trade_data.get('symbol')}:{datetime.now().isoformat()}"
            self.redis_client.setex(key, 86400, json.dumps(trade_data))  # 24 hour TTL
    
    def log_position(self, position_data: Dict[str, Any]):
        """Log position updates."""
        self.logger.info(
            "Position updated",
            symbol=position_data.get('symbol'),
            side=position_data.get('side'),
            quantity=position_data.get('quantity'),
            entry_price=position_data.get('entry_price'),
            current_price=position_data.get('current_price'),
            unrealized_pnl=position_data.get('unrealized_pnl')
        )
        
        # Update Prometheus metrics
        POSITIONS_OPEN.set(position_data.get('open_positions_count', 0))
    
    def log_portfolio(self, portfolio_data: Dict[str, Any]):
        """Log portfolio updates."""
        self.logger.info(
            "Portfolio updated",
            portfolio_value=portfolio_data.get('portfolio_value'),
            cash=portfolio_data.get('cash'),
            total_pnl=portfolio_data.get('total_pnl'),
            daily_pnl=portfolio_data.get('daily_pnl'),
            open_positions=portfolio_data.get('open_positions')
        )
        
        # Update Prometheus metrics
        PORTFOLIO_VALUE.set(portfolio_data.get('portfolio_value', 0))
        DAILY_PNL.set(portfolio_data.get('daily_pnl', 0))
    
    def log_model_performance(self, model_data: Dict[str, Any]):
        """Log model performance metrics."""
        self.logger.info(
            "Model performance",
            model_name=model_data.get('model_name'),
            accuracy=model_data.get('accuracy'),
            precision=model_data.get('precision'),
            recall=model_data.get('recall'),
            f1_score=model_data.get('f1_score'),
            training_time=model_data.get('training_time')
        )
    
    def log_error(self, error_data: Dict[str, Any]):
        """Log errors with context."""
        self.logger.error(
            "Trading bot error",
            error_type=error_data.get('error_type'),
            error_message=error_data.get('error_message'),
            component=error_data.get('component'),
            symbol=error_data.get('symbol'),
            timestamp=error_data.get('timestamp')
        )
    
    def log_risk_alert(self, alert_data: Dict[str, Any]):
        """Log risk management alerts."""
        self.logger.warning(
            "Risk alert",
            alert_type=alert_data.get('alert_type'),
            symbol=alert_data.get('symbol'),
            threshold=alert_data.get('threshold'),
            current_value=alert_data.get('current_value'),
            action_taken=alert_data.get('action_taken')
        )


class MetricsCollector:
    """Collect and expose trading metrics."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self._start_prometheus_server()
    
    def _start_prometheus_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(monitoring.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {monitoring.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    def record_signal_latency(self, latency_seconds: float):
        """Record signal generation latency."""
        SIGNAL_LATENCY.observe(latency_seconds)
    
    def record_model_prediction_time(self, prediction_time: float):
        """Record model prediction time."""
        MODEL_PREDICTION_TIME.observe(prediction_time)
    
    def update_position_count(self, count: int):
        """Update position count metric."""
        POSITIONS_OPEN.set(count)
    
    def update_portfolio_value(self, value: float):
        """Update portfolio value metric."""
        PORTFOLIO_VALUE.set(value)
    
    def update_daily_pnl(self, pnl: float):
        """Update daily P&L metric."""
        DAILY_PNL.set(pnl)


class AlertManager:
    """Manage trading alerts and notifications."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.alert_webhook_url = monitoring.alert_webhook_url
    
    async def send_alert(self, alert_type: str, message: str, data: Dict[str, Any] = None):
        """Send an alert notification."""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Log the alert
        self.logger.warning(f"Alert: {alert_type} - {message}", **data or {})
        
        # Send to webhook if configured
        if self.alert_webhook_url:
            await self._send_webhook_alert(alert)
    
    async def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send alert to webhook."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.alert_webhook_url,
                    json=alert,
                    timeout=10
                ) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to send webhook alert: {response.status}")
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    async def send_daily_summary(self, summary_data: Dict[str, Any]):
        """Send daily trading summary."""
        message = f"Daily Trading Summary - P&L: ${summary_data.get('daily_pnl', 0):.2f}"
        await self.send_alert("daily_summary", message, summary_data)
    
    async def send_risk_alert(self, risk_data: Dict[str, Any]):
        """Send risk management alert."""
        message = f"Risk Alert - {risk_data.get('alert_type', 'Unknown')}"
        await self.send_alert("risk_alert", message, risk_data)
    
    async def send_error_alert(self, error_data: Dict[str, Any]):
        """Send error alert."""
        message = f"Error Alert - {error_data.get('error_type', 'Unknown')}"
        await self.send_alert("error_alert", message, error_data)


class PerformanceMonitor:
    """Monitor system performance and latency."""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.metrics_collector = MetricsCollector()
        self.performance_data = {}
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        return datetime.now().timestamp()
    
    def end_timer(self, operation: str, start_time: float):
        """End timing an operation and record metrics."""
        end_time = datetime.now().timestamp()
        duration = end_time - start_time
        
        self.performance_data[operation] = duration
        
        # Record specific metrics
        if operation == "signal_generation":
            self.metrics_collector.record_signal_latency(duration)
        elif operation == "model_prediction":
            self.metrics_collector.record_model_prediction_time(duration)
        
        # Log if operation takes too long
        if duration > config.max_latency_ms / 1000:  # Convert ms to seconds
            self.logger.warning(
                f"Slow operation detected",
                operation=operation,
                duration_seconds=duration,
                threshold_seconds=config.max_latency_ms / 1000
            )
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        return self.performance_data.copy()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': self.get_performance_summary(),
            'issues': []
        }
        
        # Check for slow operations
        for operation, duration in self.performance_data.items():
            if duration > config.max_latency_ms / 1000:
                health_status['issues'].append(f"Slow {operation}: {duration:.3f}s")
        
        if health_status['issues']:
            health_status['status'] = 'degraded'
        
        return health_status


# Global instances
trading_logger = TradingLogger()
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
performance_monitor = PerformanceMonitor() 