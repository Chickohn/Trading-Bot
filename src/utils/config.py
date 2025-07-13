"""
Configuration management for the trading bot system.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    url: str = Field(default="postgresql://user:password@localhost:5432/trading_bot", env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    
    class Config:
        env_prefix = "DB_"


class RedisConfig(BaseSettings):
    """Redis configuration settings."""
    
    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    ttl: int = Field(default=300, env="REDIS_TTL")
    
    class Config:
        env_prefix = "REDIS_"


class AlpacaConfig(BaseSettings):
    """Alpaca trading API configuration."""
    
    api_key: str = Field(default="", env="ALPACA_API_KEY")
    secret_key: str = Field(default="", env="ALPACA_SECRET_KEY")
    base_url: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    data_url: str = Field(default="https://data.alpaca.markets", env="ALPACA_DATA_URL")
    paper_trading: bool = Field(default=True, env="PAPER_TRADING")
    
    class Config:
        env_prefix = "ALPACA_"


class PolygonConfig(BaseSettings):
    """Polygon.io market data configuration."""
    
    api_key: str = Field(default="", env="POLYGON_API_KEY")
    rate_limit: int = Field(default=5, env="POLYGON_RATE_LIMIT")
    
    class Config:
        env_prefix = "POLYGON_"


class BinanceConfig(BaseSettings):
    """Binance API configuration for crypto trading."""
    
    api_key: str = Field(default="", env="BINANCE_API_KEY")
    secret_key: str = Field(default="", env="BINANCE_SECRET_KEY")
    testnet: bool = Field(default=True, env="BINANCE_TESTNET")
    
    class Config:
        env_prefix = "BINANCE_"


class RiskConfig(BaseSettings):
    """Risk management configuration."""
    
    max_position_size: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    max_daily_loss: float = Field(default=0.05, env="MAX_DAILY_LOSS")
    stop_loss_pct: float = Field(default=0.02, env="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.04, env="TAKE_PROFIT_PCT")
    max_open_trades: int = Field(default=5, env="MAX_OPEN_TRADES")
    max_drawdown: float = Field(default=0.15, env="MAX_DRAWDOWN")
    
    @validator("max_position_size", "max_daily_loss", "stop_loss_pct", 
               "take_profit_pct", "max_drawdown")
    def validate_percentages(cls, v):
        if not 0 < v <= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v
    
    class Config:
        env_prefix = "RISK_"


class TradingConfig(BaseSettings):
    """Trading strategy configuration."""
    
    default_timeframe: str = Field(default="1m", env="DEFAULT_TIMEFRAME")
    trading_symbols: List[str] = Field(default=["AAPL", "TSLA"], env="TRADING_SYMBOLS")
    signal_cooldown: int = Field(default=300, env="SIGNAL_COOLDOWN")
    pattern_threshold: float = Field(default=0.7, env="PATTERN_DETECTION_THRESHOLD")
    
    @validator("trading_symbols", pre=True)
    def parse_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v
    
    class Config:
        env_prefix = "TRADING_"


class ModelConfig(BaseSettings):
    """ML model configuration."""
    
    update_frequency: str = Field(default="1h", env="MODEL_UPDATE_FREQUENCY")
    batch_size: int = Field(default=1000, env="BATCH_SIZE")
    model_path: str = Field(default="./models", env="MODEL_PATH")
    
    class Config:
        env_prefix = "MODEL_"


class MonitoringConfig(BaseSettings):
    """Monitoring and alerting configuration."""
    
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    alert_webhook_url: Optional[str] = Field(default=None, env="ALERT_WEBHOOK_URL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_prefix = "MONITORING_"


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration."""
    
    tracking_uri: str = Field(default="http://localhost:5000", env="MLFLOW_TRACKING_URI")
    experiment_name: str = Field(default="trading_bot_patterns", env="MLFLOW_EXPERIMENT_NAME")
    
    class Config:
        env_prefix = "MLFLOW_"


class Config(BaseSettings):
    """Main configuration class for global settings (not nested configs)."""
    
    environment: str = Field(default="development", env="ENVIRONMENT")
    # Performance settings
    max_latency_ms: int = Field(default=100, env="MAX_LATENCY_MS")
    cache_ttl: int = Field(default=300, env="CACHE_TTL")
    # Security settings
    encryption_key: str = Field(default="your_32_character_encryption_key_here", env="ENCRYPTION_KEY")
    jwt_secret: str = Field(default="your_jwt_secret_here", env="JWT_SECRET")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

# Instantiate each config as a global singleton
config = Config()
database = DatabaseConfig()
redis = RedisConfig()
alpaca = AlpacaConfig()
polygon = PolygonConfig()
binance = BinanceConfig()
risk = RiskConfig()
trading = TradingConfig()
model = ModelConfig()
monitoring = MonitoringConfig()
mlflow = MLflowConfig()

# Usage:
# from src.utils.config import config, database, redis, alpaca, polygon, binance, risk, trading, model, monitoring, mlflow 