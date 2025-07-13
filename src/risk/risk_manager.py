"""
Risk management system for trading bot.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import structlog

from signals.signal_generator import TradingSignal, SignalType
from utils.config import config, risk

logger = structlog.get_logger(__name__)


class RiskLevel(Enum):
    """Risk levels for position sizing."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class Position:
    """Trading position information."""
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L."""
        self.current_price = current_price
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def should_close(self) -> Tuple[bool, str]:
        """Check if position should be closed."""
        if self.stop_loss and self.current_price <= self.stop_loss:
            return True, "stop_loss"
        if self.take_profit and self.current_price >= self.take_profit:
            return True, "take_profit"
        return False, ""


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio."""
    total_pnl: float
    daily_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    var_95: float  # Value at Risk (95%)
    open_positions: int
    total_exposure: float
    margin_used: float
    free_margin: float


class RiskManager:
    """Main risk management system."""
    
    def __init__(self, config_section=None):
        self.config = config_section or risk
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.daily_pnl_history: List[Dict] = []
        self.risk_level = RiskLevel.MODERATE
        self.kill_switch_active = False
        self.max_daily_loss_reached = False
        
    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_info: Dict,
        volatility: float = None
    ) -> float:
        """Calculate position size based on risk management rules."""
        if self.kill_switch_active:
            return 0.0
        
        # Get account values
        portfolio_value = account_info.get('portfolio_value', 0)
        cash = account_info.get('cash', 0)
        
        if portfolio_value <= 0:
            return 0.0
        
        # Base position size as percentage of portfolio
        base_size_pct = self.config.max_position_size
        
        # Adjust for risk level
        risk_multipliers = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.5
        }
        size_pct = base_size_pct * risk_multipliers[self.risk_level]
        
        # Adjust for signal strength
        strength_multipliers = {
            "weak": 0.5,
            "medium": 0.75,
            "strong": 1.0,
            "very_strong": 1.25
        }
        size_pct *= strength_multipliers.get(signal.strength.value, 0.5)
        
        # Adjust for volatility (higher volatility = smaller position)
        if volatility:
            vol_adjustment = max(0.1, 1.0 - (volatility * 10))  # Reduce size for high volatility
            size_pct *= vol_adjustment
        
        # Calculate dollar amount
        position_value = portfolio_value * size_pct
        
        # Ensure we don't exceed available cash
        position_value = min(position_value, cash * 0.95)  # Leave 5% buffer
        
        # Calculate quantity
        quantity = position_value / signal.price
        
        # Round down to whole shares
        quantity = int(quantity)
        
        logger.info(f"Calculated position size: {quantity} shares of {signal.symbol} "
                   f"(${position_value:.2f})")
        
        return quantity
    
    def calculate_stop_loss(
        self,
        signal: TradingSignal,
        entry_price: float,
        atr: float = None
    ) -> float:
        """Calculate stop loss price."""
        if signal.signal_type == SignalType.BUY:
            # For long positions, stop loss below entry
            if atr:
                # Use ATR-based stop loss
                stop_loss = entry_price - (atr * 2)  # 2 ATR below entry
            else:
                # Use percentage-based stop loss
                stop_loss = entry_price * (1 - self.config.stop_loss_pct)
        else:
            # For short positions, stop loss above entry
            if atr:
                stop_loss = entry_price + (atr * 2)  # 2 ATR above entry
            else:
                stop_loss = entry_price * (1 + self.config.stop_loss_pct)
        
        return stop_loss
    
    def calculate_take_profit(
        self,
        signal: TradingSignal,
        entry_price: float,
        atr: float = None
    ) -> float:
        """Calculate take profit price."""
        if signal.signal_type == SignalType.BUY:
            # For long positions, take profit above entry
            if atr:
                take_profit = entry_price + (atr * 3)  # 3 ATR above entry
            else:
                take_profit = entry_price * (1 + self.config.take_profit_pct)
        else:
            # For short positions, take profit below entry
            if atr:
                take_profit = entry_price - (atr * 3)  # 3 ATR below entry
            else:
                take_profit = entry_price * (1 - self.config.take_profit_pct)
        
        return take_profit
    
    def open_position(
        self,
        signal: TradingSignal,
        quantity: float,
        entry_price: float,
        atr: float = None
    ) -> Position:
        """Open a new trading position."""
        if self.kill_switch_active:
            raise ValueError("Cannot open position - kill switch is active")
        
        if signal.symbol in self.positions:
            raise ValueError(f"Position already exists for {signal.symbol}")
        
        # Determine position side
        side = "long" if signal.signal_type == SignalType.BUY else "short"
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(signal, entry_price, atr)
        take_profit = self.calculate_take_profit(signal, entry_price, atr)
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[signal.symbol] = position
        
        logger.info(f"Opened {side} position for {signal.symbol}: "
                   f"{quantity} shares at ${entry_price:.2f}")
        
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Optional[Position]:
        """Close an existing position."""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        position.current_price = exit_price
        position.update_pnl(exit_price)
        
        # Calculate realized P&L
        if position.side == "long":
            position.realized_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            position.realized_pnl = (position.entry_price - exit_price) * position.quantity
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Closed {position.side} position for {symbol}: "
                   f"P&L = ${position.realized_pnl:.2f} ({reason})")
        
        return position
    
    def update_positions(self, market_data: Dict[str, float]):
        """Update all positions with current market prices."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.update_pnl(current_price)
                
                # Check if position should be closed
                should_close, reason = position.should_close()
                if should_close:
                    positions_to_close.append((symbol, current_price, reason))
        
        # Close positions that hit stop loss or take profit
        for symbol, price, reason in positions_to_close:
            self.close_position(symbol, price, reason)
    
    def check_risk_limits(self, account_info: Dict) -> bool:
        """Check if risk limits are exceeded."""
        # Check daily loss limit
        daily_pnl = self.calculate_daily_pnl()
        portfolio_value = account_info.get('portfolio_value', 0)
        
        if portfolio_value > 0:
            daily_loss_pct = abs(min(0, daily_pnl)) / portfolio_value
            
            if daily_loss_pct > self.config.max_daily_loss:
                logger.warning(f"Daily loss limit exceeded: {daily_loss_pct:.2%}")
                self.max_daily_loss_reached = True
                return False
        
        # Check maximum open positions
        if len(self.positions) >= self.config.max_open_trades:
            logger.warning(f"Maximum open positions reached: {len(self.positions)}")
            return False
        
        # Check maximum drawdown
        max_drawdown = self.calculate_max_drawdown()
        if max_drawdown > self.config.max_drawdown:
            logger.warning(f"Maximum drawdown exceeded: {max_drawdown:.2%}")
            return False
        
        return True
    
    def calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        # P&L from closed positions today
        for position in self.closed_positions:
            if position.entry_time.date() == today:
                daily_pnl += position.realized_pnl
        
        # Unrealized P&L from open positions
        for position in self.positions.values():
            if position.entry_time.date() == today:
                daily_pnl += position.unrealized_pnl
        
        return daily_pnl
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.daily_pnl_history:
            return 0.0
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        running_total = 0.0
        
        for entry in self.daily_pnl_history:
            running_total += entry['pnl']
            cumulative_pnl.append(running_total)
        
        if not cumulative_pnl:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_pnl[0]
        max_drawdown = 0.0
        
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            drawdown = (peak - pnl) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def get_risk_metrics(self, account_info: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        total_pnl = sum(p.realized_pnl for p in self.closed_positions)
        daily_pnl = self.calculate_daily_pnl()
        max_drawdown = self.calculate_max_drawdown()
        
        # Calculate volatility (simplified)
        if len(self.daily_pnl_history) > 1:
            pnl_values = [entry['pnl'] for entry in self.daily_pnl_history]
            volatility = np.std(pnl_values)
        else:
            volatility = 0.0
        
        # Calculate Sharpe ratio (simplified)
        if volatility > 0:
            sharpe_ratio = (daily_pnl / volatility) if volatility > 0 else 0
        else:
            sharpe_ratio = 0.0
        
        # Calculate Value at Risk (95%)
        if len(self.daily_pnl_history) > 10:
            pnl_values = [entry['pnl'] for entry in self.daily_pnl_history]
            var_95 = np.percentile(pnl_values, 5)  # 5th percentile
        else:
            var_95 = 0.0
        
        # Calculate exposure
        total_exposure = sum(
            abs(p.quantity * p.current_price) for p in self.positions.values()
        )
        
        portfolio_value = account_info.get('portfolio_value', 0)
        margin_used = total_exposure
        free_margin = portfolio_value - margin_used
        
        return RiskMetrics(
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            var_95=var_95,
            open_positions=len(self.positions),
            total_exposure=total_exposure,
            margin_used=margin_used,
            free_margin=free_margin
        )
    
    def activate_kill_switch(self, reason: str = "manual"):
        """Activate kill switch to stop all trading."""
        self.kill_switch_active = True
        logger.critical(f"Kill switch activated: {reason}")
        
        # Close all positions
        symbols_to_close = list(self.positions.keys())
        for symbol in symbols_to_close:
            # Use current price or estimate
            current_price = self.positions[symbol].current_price
            self.close_position(symbol, current_price, "kill_switch")
    
    def deactivate_kill_switch(self):
        """Deactivate kill switch."""
        self.kill_switch_active = False
        logger.info("Kill switch deactivated")
    
    def reset_daily_limits(self):
        """Reset daily limits (call at start of new day)."""
        self.max_daily_loss_reached = False
        logger.info("Daily limits reset")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of all positions."""
        summary = {
            'open_positions': len(self.positions),
            'total_unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'positions': {}
        }
        
        for symbol, position in self.positions.items():
            summary['positions'][symbol] = {
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit
            }
        
        return summary 