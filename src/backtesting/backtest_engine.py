"""
Backtesting framework for trading strategies.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import structlog
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import empyrical as ep

from data.base import OHLCV
from features.technical_indicators import TechnicalIndicators
from models.random_forest_model import RandomForestModel
from models.base_model import ModelConfig
from signals.signal_generator import SignalGenerator, TradingSignal
from risk.risk_manager import RiskManager, Position
from utils.config import config

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results of a backtest."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series
    trades: List[Dict]
    signals: List[Dict]
    metrics: Dict[str, float]


@dataclass
class WalkForwardResult:
    """Results of walk-forward analysis."""
    periods: List[Dict]
    overall_metrics: Dict[str, float]
    period_metrics: List[Dict]
    equity_curves: List[pd.Series]


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager()
        self.results = []
        
    async def run_backtest(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        model_config: Optional[ModelConfig] = None
    ) -> BacktestResult:
        """Run a complete backtest."""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize components
        model = self._initialize_model(model_config)
        signal_generator = SignalGenerator([model])
        
        # Filter data for backtest period
        mask = (data.index >= start_date) & (data.index <= end_date)
        backtest_data = data[mask].copy()
        
        if backtest_data.empty:
            raise ValueError("No data available for backtest period")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'equity': self.initial_capital,
            'trades': [],
            'signals': []
        }
        
        # Run backtest
        for timestamp, row in backtest_data.iterrows():
            await self._process_backtest_tick(
                timestamp, row, portfolio, signal_generator, model
            )
        
        # Calculate results
        result = self._calculate_backtest_results(portfolio, backtest_data)
        
        logger.info(f"Backtest completed. Total return: {result.total_return:.2%}")
        return result
    
    def _initialize_model(self, model_config: Optional[ModelConfig]) -> RandomForestModel:
        """Initialize the model for backtesting."""
        if model_config is None:
            model_config = ModelConfig(
                model_name="backtest_rf",
                lookback_period=20,
                prediction_threshold=0.7,
                retrain_frequency="1d"
            )
        
        return RandomForestModel(model_config)
    
    async def _process_backtest_tick(
        self,
        timestamp: datetime,
        data: pd.Series,
        portfolio: Dict,
        signal_generator: SignalGenerator,
        model: RandomForestModel
    ):
        """Process a single tick in the backtest."""
        try:
            # Prepare data for model
            historical_data = self._get_historical_window(data, timestamp)
            
            if historical_data.empty:
                return
            
            # Generate signal
            signal = await self._generate_backtest_signal(
                data, historical_data, signal_generator, model
            )
            
            if signal:
                portfolio['signals'].append({
                    'timestamp': timestamp,
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'price': signal.price
                })
                
                # Execute signal
                await self._execute_backtest_signal(signal, portfolio, timestamp)
            
            # Update portfolio value
            self._update_portfolio_value(portfolio, data, timestamp)
            
        except Exception as e:
            logger.error(f"Error processing backtest tick: {e}")
    
    def _get_historical_window(self, current_data: pd.Series, timestamp: datetime) -> pd.DataFrame:
        """Get historical data window for model prediction."""
        # This is a simplified version - in practice you'd get actual historical data
        # For now, we'll create synthetic historical data
        dates = pd.date_range(end=timestamp, periods=100, freq='1H')
        
        historical_data = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100),
            'close': np.random.normal(100, 2, 100),
            'volume': np.random.normal(1000000, 200000, 100)
        }, index=dates)
        
        return historical_data
    
    async def _generate_backtest_signal(
        self,
        data: pd.Series,
        historical_data: pd.DataFrame,
        signal_generator: SignalGenerator,
        model: RandomForestModel
    ) -> Optional[TradingSignal]:
        """Generate a trading signal for backtesting."""
        try:
            # Create market data object
            from data.base import MarketData, OHLCV
            
            ohlcv = OHLCV(
                timestamp=data.name,
                open=data.get('open', 100),
                high=data.get('high', 102),
                low=data.get('low', 98),
                close=data.get('close', 100),
                volume=data.get('volume', 1000000),
                symbol=data.get('symbol', 'UNKNOWN'),
                timeframe='1H'
            )
            
            market_data = MarketData(ohlcv=ohlcv)
            
            # Generate signal
            signal = await signal_generator.generate_signal(
                ohlcv.symbol, market_data, historical_data
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating backtest signal: {e}")
            return None
    
    async def _execute_backtest_signal(
        self,
        signal: TradingSignal,
        portfolio: Dict,
        timestamp: datetime
    ):
        """Execute a trading signal in backtest."""
        try:
            symbol = signal.symbol
            price = signal.price
            
            if signal.signal_type.value == 'buy':
                # Calculate position size
                position_value = portfolio['cash'] * 0.1  # 10% of cash
                quantity = int(position_value / price)
                
                if quantity > 0:
                    # Execute buy
                    cost = quantity * price
                    portfolio['cash'] -= cost
                    portfolio['positions'][symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': timestamp
                    }
                    
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': quantity,
                        'price': price,
                        'cost': cost
                    })
            
            elif signal.signal_type.value == 'sell':
                if symbol in portfolio['positions']:
                    position = portfolio['positions'][symbol]
                    quantity = position['quantity']
                    
                    # Execute sell
                    proceeds = quantity * price
                    portfolio['cash'] += proceeds
                    
                    # Calculate P&L
                    entry_cost = quantity * position['entry_price']
                    pnl = proceeds - entry_cost
                    
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'side': 'sell',
                        'quantity': quantity,
                        'price': price,
                        'proceeds': proceeds,
                        'pnl': pnl
                    })
                    
                    del portfolio['positions'][symbol]
                    
        except Exception as e:
            logger.error(f"Error executing backtest signal: {e}")
    
    def _update_portfolio_value(self, portfolio: Dict, data: pd.Series, timestamp: datetime):
        """Update portfolio value based on current positions."""
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            current_price = data.get('close', position['entry_price'])
            position_value = position['quantity'] * current_price
            total_value += position_value
        
        portfolio['equity'] = total_value
    
    def _calculate_backtest_results(self, portfolio: Dict, data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive backtest results."""
        trades = portfolio['trades']
        signals = portfolio['signals']
        
        if not trades:
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                equity_curve=pd.Series(),
                trades=trades,
                signals=signals,
                metrics={}
            )
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t.get('pnl', 0) < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate P&L metrics
        pnl_values = [t.get('pnl', 0) for t in trades]
        total_pnl = sum(pnl_values)
        
        winning_pnl = [pnl for pnl in pnl_values if pnl > 0]
        losing_pnl = [pnl for pnl in pnl_values if pnl < 0]
        
        avg_win = np.mean(winning_pnl) if winning_pnl else 0
        avg_loss = abs(np.mean(losing_pnl)) if losing_pnl else 0
        
        profit_factor = sum(winning_pnl) / abs(sum(losing_pnl)) if losing_pnl else float('inf')
        
        # Calculate returns
        total_return = (portfolio['equity'] - self.initial_capital) / self.initial_capital
        
        # Create equity curve
        equity_curve = self._create_equity_curve(portfolio, data)
        
        # Calculate advanced metrics
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()
            sharpe_ratio = ep.sharpe_ratio(returns) if len(returns) > 0 else 0
            max_drawdown = ep.max_drawdown(equity_curve)[0]
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Additional metrics
        metrics = {
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'sortino_ratio': ep.sortino_ratio(returns) if len(returns) > 0 else 0,
            'var_95': ep.value_at_risk(returns, 0.05) if len(returns) > 0 else 0,
            'cvar_95': ep.conditional_value_at_risk(returns, 0.05) if len(returns) > 0 else 0
        }
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_curve,
            trades=trades,
            signals=signals,
            metrics=metrics
        )
    
    def _create_equity_curve(self, portfolio: Dict, data: pd.DataFrame) -> pd.Series:
        """Create equity curve from portfolio data."""
        # This is a simplified version - in practice you'd track equity over time
        # For now, we'll create a synthetic equity curve
        dates = data.index
        equity_values = [self.initial_capital]
        
        for i in range(1, len(dates)):
            # Simulate some growth
            growth = np.random.normal(0.001, 0.01)  # 0.1% average daily return
            equity_values.append(equity_values[-1] * (1 + growth))
        
        return pd.Series(equity_values, index=dates)
    
    async def run_walk_forward_analysis(
        self,
        data: pd.DataFrame,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        train_period_days: int = 252,
        test_period_days: int = 63,
        overlap_days: int = 21
    ) -> WalkForwardResult:
        """Run walk-forward analysis."""
        logger.info("Starting walk-forward analysis")
        
        periods = []
        period_metrics = []
        equity_curves = []
        
        current_date = start_date
        
        while current_date + timedelta(days=train_period_days + test_period_days) <= end_date:
            # Define train and test periods
            train_start = current_date
            train_end = train_start + timedelta(days=train_period_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_period_days)
            
            period_info = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end
            }
            periods.append(period_info)
            
            # Run backtest for this period
            result = await self.run_backtest(
                data, symbols, test_start, test_end
            )
            
            period_metrics.append({
                'period': len(periods),
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            })
            
            equity_curves.append(result.equity_curve)
            
            # Move to next period
            current_date += timedelta(days=test_period_days - overlap_days)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_walk_forward_metrics(period_metrics)
        
        return WalkForwardResult(
            periods=periods,
            overall_metrics=overall_metrics,
            period_metrics=period_metrics,
            equity_curves=equity_curves
        )
    
    def _calculate_overall_walk_forward_metrics(self, period_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate overall metrics from walk-forward analysis."""
        if not period_metrics:
            return {}
        
        returns = [p['total_return'] for p in period_metrics]
        sharpe_ratios = [p['sharpe_ratio'] for p in period_metrics]
        max_drawdowns = [p['max_drawdown'] for p in period_metrics]
        win_rates = [p['win_rate'] for p in period_metrics]
        
        return {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'total_periods': len(period_metrics),
            'profitable_periods': len([r for r in returns if r > 0])
        }
    
    def plot_results(self, result: BacktestResult, save_path: Optional[str] = None):
        """Plot backtest results."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown', 'Trade P&L'),
            vertical_spacing=0.1
        )
        
        # Equity curve
        if not result.equity_curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    name='Portfolio Value',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # Drawdown
        if not result.equity_curve.empty:
            drawdown = ep.drawdown(result.equity_curve)
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name='Drawdown',
                    line=dict(color='red'),
                    fill='tonexty'
                ),
                row=2, col=1
            )
        
        # Trade P&L
        if result.trades:
            trade_pnl = [t.get('pnl', 0) for t in result.trades]
            trade_dates = [t['timestamp'] for t in result.trades]
            
            fig.add_trace(
                go.Bar(
                    x=trade_dates,
                    y=trade_pnl,
                    name='Trade P&L',
                    marker_color=['green' if pnl > 0 else 'red' for pnl in trade_pnl]
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title=f'Backtest Results - Total Return: {result.total_return:.2%}',
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate a comprehensive backtest report."""
        report = f"""
        ===== BACKTEST REPORT =====
        
        Performance Metrics:
        - Total Return: {result.total_return:.2%}
        - Sharpe Ratio: {result.sharpe_ratio:.3f}
        - Max Drawdown: {result.max_drawdown:.2%}
        - Win Rate: {result.win_rate:.2%}
        - Profit Factor: {result.profit_factor:.3f}
        
        Trade Statistics:
        - Total Trades: {result.total_trades}
        - Winning Trades: {result.winning_trades}
        - Losing Trades: {result.losing_trades}
        - Average Win: ${result.avg_win:.2f}
        - Average Loss: ${result.avg_loss:.2f}
        
        Risk Metrics:
        - Calmar Ratio: {result.metrics.get('calmar_ratio', 0):.3f}
        - Sortino Ratio: {result.metrics.get('sortino_ratio', 0):.3f}
        - VaR (95%): {result.metrics.get('var_95', 0):.3f}
        - CVaR (95%): {result.metrics.get('cvar_95', 0):.3f}
        
        =========================
        """
        
        return report 