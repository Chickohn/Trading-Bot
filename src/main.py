"""
Main trading bot orchestrator.
"""

import asyncio
import signal
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import structlog
from dataclasses import dataclass
import click

# Ensure src is on the Python path
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils.config import config, trading
from data.base import DataManager, OHLCV
from data.alpaca_provider import AlpacaDataProvider
from features.technical_indicators import TechnicalIndicators
from models.base_model import ModelConfig
from models.random_forest_model import RandomForestModel
from signals.signal_generator import SignalGenerator, SignalValidator
from risk.risk_manager import RiskManager
from monitoring.logger import setup_logging

logger = structlog.get_logger(__name__)


@dataclass
class TradingBotState:
    """State of the trading bot."""
    is_running: bool = False
    is_paper_trading: bool = True
    symbols: List[str] = None
    data_manager: DataManager = None
    models: List = None
    signal_generator: SignalGenerator = None
    risk_manager: RiskManager = None
    signal_validator: SignalValidator = None
    indicators: TechnicalIndicators = None


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, mode: str = "paper"):
        self.state = TradingBotState()
        self.state.is_paper_trading = mode == "paper"
        self.state.symbols = trading.trading_symbols
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup logging
        setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_components(self):
        """Initialize all trading bot components."""
        logger.info("Initializing trading bot components...")
        
        # Initialize data manager
        self.state.data_manager = DataManager()
        
        # Add data providers
        alpaca_provider = AlpacaDataProvider()
        self.state.data_manager.add_provider("alpaca", alpaca_provider)
        
        # Initialize technical indicators
        self.state.indicators = TechnicalIndicators()
        
        # Initialize models
        self._initialize_models()
        
        # Initialize signal generator
        self.state.signal_generator = SignalGenerator(self.state.models)
        
        # Initialize risk manager
        self.state.risk_manager = RiskManager()
        
        # Initialize signal validator
        self.state.signal_validator = SignalValidator()
        
        logger.info("All components initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models."""
        self.state.models = []
        
        # Random Forest model
        rf_config = ModelConfig(
            model_name="random_forest",
            lookback_period=20,
            prediction_threshold=0.7,
            retrain_frequency="1d"
        )
        
        rf_model = RandomForestModel(rf_config)
        self.state.models.append(rf_model)
        
        logger.info(f"Initialized {len(self.state.models)} models")
    
    async def start(self):
        """Start the trading bot."""
        if self.running:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot...")
        self.running = True
        
        try:
            # Connect to data providers
            await self.state.data_manager.connect_all()
            
            # Train models if needed
            await self._train_models()
            
            # Start real-time data streams
            await self._start_data_streams()
            
            # Main trading loop
            await self._trading_loop()
            
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot."""
        if not self.running:
            return
        
        logger.info("Stopping trading bot...")
        self.running = False
        self.shutdown_event.set()
        
        # Stop data streams
        await self.state.data_manager.stop_all_streams()
        
        # Disconnect from data providers
        await self.state.data_manager.disconnect_all()
        
        logger.info("Trading bot stopped")
    
    async def _train_models(self):
        """Train all models with historical data."""
        logger.info("Training models...")
        
        for model in self.state.models:
            try:
                # Get historical data for training
                training_data = await self._get_training_data()
                
                if training_data.empty:
                    logger.warning(f"No training data available for {model.config.model_name}")
                    continue
                
                # Prepare features and target
                X = model.prepare_features(training_data)
                y = model.prepare_target(training_data)
                
                # Align X and y
                common_index = X.index.intersection(y.index)
                X = X.loc[common_index]
                y = y.loc[common_index]
                
                if len(X) < 100:
                    logger.warning(f"Insufficient training data for {model.config.model_name}")
                    continue
                
                # Train model
                metrics = model.train(X, y)
                logger.info(f"Trained {model.config.model_name} with accuracy: {metrics.get('accuracy', 0):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model.config.model_name}: {e}")
    
    async def _get_training_data(self) -> pd.DataFrame:
        """Get historical data for model training."""
        # Get data for the first symbol (assuming similar patterns across symbols)
        symbol = self.state.symbols[0]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        ohlcv_data = await self.state.data_manager.get_data_from_provider(
            "alpaca", symbol, "1h", start_date, end_date
        )
        
        if not ohlcv_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            }
            for data in ohlcv_data
        ])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _start_data_streams(self):
        """Start real-time data streams for all symbols."""
        logger.info("Starting real-time data streams...")
        
        for symbol in self.state.symbols:
            await self.state.data_manager.start_realtime_stream(
                "alpaca",
                [symbol],
                trading.default_timeframe,
                self._process_market_data
            )
    
    async def _process_market_data(self, market_data):
        """Process incoming market data."""
        try:
            symbol = market_data.ohlcv.symbol
            
            # Update risk manager positions
            self.state.risk_manager.update_positions({symbol: market_data.ohlcv.close})
            
            # Get historical data for analysis
            historical_data = await self._get_recent_data(symbol)
            
            if historical_data.empty:
                return
            
            # Generate trading signal
            signal = await self.state.signal_generator.generate_signal(
                symbol, market_data, historical_data
            )
            
            if signal:
                # Validate signal
                if not self.state.signal_validator.validate_signal(signal, market_data):
                    logger.warning(f"Signal validation failed for {symbol}")
                    return
                
                # Get account info
                account_info = await self._get_account_info()
                
                # Check risk limits
                if not self.state.risk_manager.check_risk_limits(account_info):
                    logger.warning(f"Risk limits exceeded for {symbol}")
                    return
                
                # Check if signal should be executed
                if not self.state.signal_validator.should_execute_signal(signal, account_info):
                    logger.warning(f"Signal execution blocked for {symbol}")
                    return
                
                # Execute signal
                await self._execute_signal(signal, market_data, account_info)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def _get_recent_data(self, symbol: str) -> pd.DataFrame:
        """Get recent historical data for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        ohlcv_data = await self.state.data_manager.get_data_from_provider(
            "alpaca", symbol, trading.default_timeframe, start_date, end_date
        )
        
        if not ohlcv_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': data.timestamp,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            }
            for data in ohlcv_data
        ])
        
        df.set_index('timestamp', inplace=True)
        return df
    
    async def _get_account_info(self) -> Dict:
        """Get current account information."""
        try:
            # Get account info from Alpaca
            alpaca_provider = self.state.data_manager.providers.get("alpaca")
            if alpaca_provider:
                return await alpaca_provider.get_account_info()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        
        # Return default values if provider not available
        return {
            'cash': 10000,
            'portfolio_value': 10000,
            'buying_power': 10000,
            'equity': 10000
        }
    
    async def _execute_signal(self, signal, market_data, account_info):
        """Execute a trading signal."""
        try:
            if self.state.is_paper_trading:
                await self._execute_paper_trade(signal, market_data, account_info)
            else:
                await self._execute_live_trade(signal, market_data, account_info)
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _execute_paper_trade(self, signal, market_data, account_info):
        """Execute a paper trade."""
        # Calculate position size
        quantity = self.state.risk_manager.calculate_position_size(
            signal, account_info
        )
        
        if quantity <= 0:
            logger.info(f"Position size too small for {signal.symbol}")
            return
        
        # Open position in risk manager
        position = self.state.risk_manager.open_position(
            signal, quantity, signal.price
        )
        
        logger.info(f"Paper trade executed: {signal.signal_type.value} "
                   f"{quantity} shares of {signal.symbol} at ${signal.price:.2f}")
    
    async def _execute_live_trade(self, signal, market_data, account_info):
        """Execute a live trade."""
        # Calculate position size
        quantity = self.state.risk_manager.calculate_position_size(
            signal, account_info
        )
        
        if quantity <= 0:
            logger.info(f"Position size too small for {signal.symbol}")
            return
        
        # Place order with broker
        alpaca_provider = self.state.data_manager.providers.get("alpaca")
        if alpaca_provider:
            side = "buy" if signal.signal_type.value == "buy" else "sell"
            
            order_result = await alpaca_provider.place_order(
                signal.symbol, side, quantity
            )
            
            if order_result:
                # Open position in risk manager
                position = self.state.risk_manager.open_position(
                    signal, quantity, signal.price
                )
                
                logger.info(f"Live trade executed: {order_result}")
    
    async def _trading_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
                
                # Periodic tasks
                await self._periodic_tasks()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _periodic_tasks(self):
        """Execute periodic tasks."""
        now = datetime.now()
        
        # Check if models need retraining (every hour)
        if now.minute == 0:
            for model in self.state.models:
                if model.needs_retraining():
                    logger.info(f"Retraining model: {model.config.model_name}")
                    await self._train_models()
                    break
        
        # Log status (every 5 minutes)
        if now.minute % 5 == 0 and now.second == 0:
            await self._log_status()
    
    async def _log_status(self):
        """Log current trading bot status."""
        try:
            # Get account info
            account_info = await self._get_account_info()
            
            # Get risk metrics
            risk_metrics = self.state.risk_manager.get_risk_metrics(account_info)
            
            # Get signal statistics
            signal_stats = self.state.signal_generator.get_signal_statistics()
            
            logger.info("Trading Bot Status", extra={
                'portfolio_value': account_info.get('portfolio_value', 0),
                'cash': account_info.get('cash', 0),
                'total_pnl': risk_metrics.total_pnl,
                'daily_pnl': risk_metrics.daily_pnl,
                'open_positions': risk_metrics.open_positions,
                'total_signals': signal_stats.get('total_signals', 0),
                'average_confidence': signal_stats.get('average_confidence', 0)
            })
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())


@click.command()
@click.option('--mode', default='paper', type=click.Choice(['paper', 'live']), 
              help='Trading mode')
@click.option('--symbols', help='Comma-separated list of symbols to trade')
@click.option('--config-file', help='Path to configuration file')
def main(mode, symbols, config_file):
    """Trading Bot CLI."""
    try:
        # Update symbols if provided
        if symbols:
            trading.trading_symbols = [s.strip() for s in symbols.split(',')]
        
        # Create and start trading bot
        bot = TradingBot(mode=mode)
        
        # Run the bot
        asyncio.run(bot.start())
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 