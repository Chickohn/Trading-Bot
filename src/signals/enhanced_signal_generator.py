#!/usr/bin/env python3
"""
Enhanced Signal Generator with Multi-Timeframe Analysis
Generates high-quality trading signals using advanced techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

@dataclass
class EnhancedSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0-1
    confidence: float  # 0-1
    timeframe: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    signals_confluence: int  # Number of confirming signals
    market_regime: str  # Current market regime
    factor_exposure: Dict[str, float]  # Factor loadings

class EnhancedSignalGenerator:
    """
    Advanced signal generation using:
    1. Multi-timeframe analysis
    2. Factor-based signals (momentum, mean reversion, quality, etc.)
    3. Market regime adaptation
    4. Signal confluence and filtering
    5. Risk-adjusted entry/exit points
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.signal_history = {}
        self.factor_models = self._initialize_factor_models()
    
    def _get_default_config(self) -> Dict:
        """Default configuration for signal generation."""
        return {
            'timeframes': ['15m', '1h', '4h', '1d'],
            'primary_timeframe': '1h',
            'signal_threshold': 0.6,
            'confluence_requirement': 2,  # Minimum signals for confirmation
            'max_risk_per_trade': 0.02,  # 2% max risk per trade
            'min_risk_reward': 2.0,  # Minimum 2:1 risk/reward
            'lookback_periods': {
                '15m': 96,   # 1 day
                '1h': 168,   # 1 week  
                '4h': 180,   # 1 month
                '1d': 252    # 1 year
            }
        }
    
    def _initialize_factor_models(self) -> Dict:
        """Initialize factor-based signal models."""
        return {
            'momentum': self._momentum_factor,
            'mean_reversion': self._mean_reversion_factor,
            'quality': self._quality_factor,
            'volatility': self._volatility_factor,
            'volume': self._volume_factor,
            'pattern': self._pattern_factor
        }
    
    async def generate_enhanced_signals(
        self,
        market_data: Dict[str, pd.DataFrame],  # Multiple timeframes
        technical_indicators: Dict[str, pd.DataFrame],
        market_regime: str,
        symbols: List[str]
    ) -> List[EnhancedSignal]:
        """Generate enhanced trading signals with multi-timeframe analysis."""
        
        enhanced_signals = []
        
        for symbol in symbols:
            try:
                # Get multi-timeframe data
                symbol_data = {tf: data for tf, data in market_data.items() 
                              if symbol in data.columns or any(symbol in str(col) for col in data.columns)}
                
                if not symbol_data:
                    continue
                
                # Generate signals for each timeframe
                timeframe_signals = {}
                for timeframe in self.config['timeframes']:
                    if timeframe in symbol_data:
                        signals = await self._generate_timeframe_signals(
                            symbol, timeframe, symbol_data[timeframe],
                            technical_indicators.get(f"{symbol}_{timeframe}"),
                            market_regime
                        )
                        timeframe_signals[timeframe] = signals
                
                # Apply signal confluence and filtering
                confluent_signals = self._apply_signal_confluence(
                    symbol, timeframe_signals, market_regime
                )
                
                enhanced_signals.extend(confluent_signals)
                
            except Exception as e:
                print(f"Error generating signals for {symbol}: {e}")
                continue
        
        # Rank and filter signals
        ranked_signals = self._rank_and_filter_signals(enhanced_signals, market_regime)
        
        return ranked_signals
    
    async def _generate_timeframe_signals(
        self,
        symbol: str,
        timeframe: str,
        price_data: pd.DataFrame,
        indicators: Optional[pd.DataFrame],
        market_regime: str
    ) -> List[Dict]:
        """Generate signals for a specific timeframe."""
        
        signals = []
        
        if price_data.empty:
            return signals
        
        # Factor-based signal generation
        factor_signals = {}
        for factor_name, factor_func in self.factor_models.items():
            try:
                factor_signal = factor_func(price_data, indicators, timeframe)
                factor_signals[factor_name] = factor_signal
            except Exception as e:
                print(f"Error calculating {factor_name} factor for {symbol}: {e}")
                factor_signals[factor_name] = {'signal': 0, 'strength': 0, 'confidence': 0}
        
        # Combine factor signals based on market regime
        combined_signal = self._combine_factor_signals(factor_signals, market_regime)
        
        if abs(combined_signal['signal']) > self.config['signal_threshold']:
            # Calculate entry/exit levels
            entry_exit = self._calculate_entry_exit_levels(
                price_data, combined_signal, timeframe
            )
            
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': 'BUY' if combined_signal['signal'] > 0 else 'SELL',
                'strength': abs(combined_signal['signal']),
                'confidence': combined_signal['confidence'],
                'entry_price': entry_exit['entry'],
                'stop_loss': entry_exit['stop_loss'],
                'take_profit': entry_exit['take_profit'],
                'risk_reward_ratio': entry_exit['risk_reward'],
                'factor_signals': factor_signals,
                'market_regime': market_regime
            }
            
            signals.append(signal)
        
        return signals
    
    def _momentum_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate momentum factor signal."""
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        
        # Multi-period momentum
        periods = [5, 10, 21, 63] if timeframe in ['1h', '4h'] else [10, 21, 63, 126]
        momentum_scores = []
        
        for period in periods:
            if len(close_prices) >= period + 1:
                momentum = (close_prices.iloc[-1] - close_prices.iloc[-(period+1)]) / close_prices.iloc[-(period+1)]
                momentum_scores.append(momentum)
        
        if momentum_scores:
            # Weighted average (more weight on longer periods)
            weights = np.array([1, 2, 3, 4][:len(momentum_scores)])
            weights = weights / weights.sum()
            
            avg_momentum = np.average(momentum_scores, weights=weights)
            
            # Momentum strength and confidence
            strength = min(abs(avg_momentum) * 10, 1.0)  # Scale to 0-1
            confidence = min(len(momentum_scores) / 4, 1.0)  # More periods = higher confidence
            
            signal = np.tanh(avg_momentum * 5)  # Normalize to -1 to 1
            
            return {
                'signal': signal,
                'strength': strength,
                'confidence': confidence,
                'momentum_scores': momentum_scores
            }
        
        return {'signal': 0, 'strength': 0, 'confidence': 0}
    
    def _mean_reversion_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate mean reversion factor signal."""
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        
        # Bollinger Bands mean reversion
        if indicators is not None and 'bb_upper' in indicators.columns:
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            bb_middle = indicators['bb_middle'].iloc[-1]
            current_price = close_prices.iloc[-1]
            
            # Position within bands
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            # Mean reversion signal (inverse of position)
            if bb_position > 0.8:  # Near upper band - sell signal
                signal = -0.8
                strength = min((bb_position - 0.8) * 5, 1.0)
            elif bb_position < 0.2:  # Near lower band - buy signal
                signal = 0.8
                strength = min((0.2 - bb_position) * 5, 1.0)
            else:
                signal = 0
                strength = 0
            
            confidence = 0.8  # High confidence in BB mean reversion
            
        else:
            # RSI mean reversion
            if len(close_prices) >= 14:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                if current_rsi > 70:  # Overbought - sell signal
                    signal = -(current_rsi - 70) / 30
                    strength = min((current_rsi - 70) / 30, 1.0)
                elif current_rsi < 30:  # Oversold - buy signal
                    signal = (30 - current_rsi) / 30
                    strength = min((30 - current_rsi) / 30, 1.0)
                else:
                    signal = 0
                    strength = 0
                
                confidence = 0.6  # Moderate confidence in RSI mean reversion
            else:
                signal = 0
                strength = 0
                confidence = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence
        }
    
    def _quality_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate quality factor signal (trend consistency, volatility)."""
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        
        # Trend consistency (lower volatility = higher quality)
        if len(close_prices) >= 21:
            returns = close_prices.pct_change().dropna()
            volatility = returns.tail(21).std()
            
            # Moving average consistency
            ma_20 = close_prices.rolling(20).mean()
            if len(ma_20.dropna()) >= 5:
                ma_slope = (ma_20.iloc[-1] - ma_20.iloc[-5]) / ma_20.iloc[-5]
                
                # Quality signal favors trending with low volatility
                trend_quality = abs(ma_slope) / (volatility + 0.001)  # Avoid division by zero
                
                signal = np.tanh(ma_slope * 10) * min(trend_quality * 2, 1.0)
                strength = min(trend_quality, 1.0)
                confidence = 0.7
            else:
                signal = 0
                strength = 0
                confidence = 0
        else:
            signal = 0
            strength = 0
            confidence = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence
        }
    
    def _volatility_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate volatility factor signal."""
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        
        if len(close_prices) >= 21:
            returns = close_prices.pct_change().dropna()
            current_vol = returns.tail(21).std()
            historical_vol = returns.std()
            
            # Volatility regime signal
            vol_ratio = current_vol / (historical_vol + 0.001)
            
            # In volatile periods, fade momentum; in low vol, follow trends
            if vol_ratio > 1.5:  # High volatility - mean reversion
                signal = -0.5
                strength = min((vol_ratio - 1.5) * 2, 1.0)
            elif vol_ratio < 0.7:  # Low volatility - momentum
                signal = 0.5
                strength = min((0.7 - vol_ratio) * 2, 1.0)
            else:
                signal = 0
                strength = 0
            
            confidence = 0.6
        else:
            signal = 0
            strength = 0
            confidence = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence
        }
    
    def _volume_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate volume factor signal."""
        
        if 'volume' not in price_data.columns:
            return {'signal': 0, 'strength': 0, 'confidence': 0}
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        volume = price_data['volume']
        
        if len(volume) >= 21:
            # Volume confirmation signal
            price_change = close_prices.pct_change().iloc[-1]
            volume_ratio = volume.iloc[-1] / volume.rolling(21).mean().iloc[-1]
            
            # Strong volume + price move = confirmation
            if abs(price_change) > 0.02 and volume_ratio > 1.5:  # 2% price move + 50% volume spike
                signal = np.sign(price_change) * min(volume_ratio / 3, 1.0)
                strength = min(volume_ratio / 2, 1.0)
                confidence = 0.8
            else:
                signal = 0
                strength = 0
                confidence = 0
        else:
            signal = 0
            strength = 0
            confidence = 0
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence
        }
    
    def _pattern_factor(self, price_data: pd.DataFrame, indicators: Optional[pd.DataFrame], timeframe: str) -> Dict:
        """Calculate pattern recognition factor signal."""
        
        if len(price_data) < 20:
            return {'signal': 0, 'strength': 0, 'confidence': 0}
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        
        # Simple pattern recognition
        signal = 0
        strength = 0
        confidence = 0
        
        # Support/Resistance breakout
        recent_high = close_prices.tail(20).max()
        recent_low = close_prices.tail(20).min()
        current_price = close_prices.iloc[-1]
        
        # Breakout signals
        if current_price > recent_high * 1.01:  # 1% above recent high
            signal = 0.7
            strength = 0.8
            confidence = 0.7
        elif current_price < recent_low * 0.99:  # 1% below recent low
            signal = -0.7
            strength = 0.8
            confidence = 0.7
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence
        }
    
    def _combine_factor_signals(self, factor_signals: Dict, market_regime: str) -> Dict:
        """Combine factor signals based on market regime."""
        
        # Market regime weights
        regime_weights = {
            'bull': {
                'momentum': 0.4, 'mean_reversion': 0.1, 'quality': 0.2,
                'volatility': 0.1, 'volume': 0.1, 'pattern': 0.1
            },
            'bear': {
                'momentum': 0.2, 'mean_reversion': 0.3, 'quality': 0.2,
                'volatility': 0.2, 'volume': 0.05, 'pattern': 0.05
            },
            'sideways': {
                'momentum': 0.1, 'mean_reversion': 0.4, 'quality': 0.2,
                'volatility': 0.15, 'volume': 0.1, 'pattern': 0.05
            },
            'volatile': {
                'momentum': 0.15, 'mean_reversion': 0.35, 'quality': 0.1,
                'volatility': 0.25, 'volume': 0.1, 'pattern': 0.05
            }
        }
        
        weights = regime_weights.get(market_regime, regime_weights['sideways'])
        
        # Weighted combination
        combined_signal = 0
        combined_strength = 0
        combined_confidence = 0
        
        for factor, weight in weights.items():
            if factor in factor_signals:
                factor_data = factor_signals[factor]
                combined_signal += factor_data['signal'] * weight
                combined_strength += factor_data['strength'] * weight
                combined_confidence += factor_data['confidence'] * weight
        
        return {
            'signal': np.tanh(combined_signal),  # Normalize to -1 to 1
            'strength': min(combined_strength, 1.0),
            'confidence': min(combined_confidence, 1.0)
        }
    
    def _calculate_entry_exit_levels(self, price_data: pd.DataFrame, signal: Dict, timeframe: str) -> Dict:
        """Calculate optimal entry, stop loss, and take profit levels."""
        
        close_prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, -1]
        current_price = close_prices.iloc[-1]
        
        # Calculate ATR for dynamic stop losses
        if all(col in price_data.columns for col in ['high', 'low']):
            high_prices = price_data['high']
            low_prices = price_data['low']
            
            tr = np.maximum(
                high_prices - low_prices,
                np.maximum(
                    abs(high_prices - close_prices.shift(1)),
                    abs(low_prices - close_prices.shift(1))
                )
            )
            
            if len(tr.dropna()) >= 14:
                atr = tr.rolling(14).mean().iloc[-1]
            else:
                atr = current_price * 0.02  # 2% default
        else:
            # Use price volatility as proxy
            returns = close_prices.pct_change().dropna()
            if len(returns) >= 14:
                atr = returns.rolling(14).std().iloc[-1] * current_price * np.sqrt(14)
            else:
                atr = current_price * 0.02
        
        # Dynamic stop loss and take profit based on signal strength and timeframe
        timeframe_multipliers = {
            '15m': 1.0,
            '1h': 1.5,
            '4h': 2.0,
            '1d': 3.0
        }
        
        multiplier = timeframe_multipliers.get(timeframe, 1.5)
        
        if signal['signal'] > 0:  # Buy signal
            entry_price = current_price
            stop_loss = current_price - (atr * multiplier * 1.5)
            take_profit = current_price + (atr * multiplier * 3.0)  # 2:1 R/R minimum
        else:  # Sell signal
            entry_price = current_price
            stop_loss = current_price + (atr * multiplier * 1.5)
            take_profit = current_price - (atr * multiplier * 3.0)
        
        # Calculate risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward': risk_reward_ratio
        }
    
    def _apply_signal_confluence(self, symbol: str, timeframe_signals: Dict, market_regime: str) -> List[EnhancedSignal]:
        """Apply signal confluence across timeframes."""
        
        confluent_signals = []
        primary_tf = self.config['primary_timeframe']
        
        if primary_tf not in timeframe_signals or not timeframe_signals[primary_tf]:
            return confluent_signals
        
        for primary_signal in timeframe_signals[primary_tf]:
            # Count confirming signals from other timeframes
            confirming_signals = 0
            signal_strength_sum = primary_signal['strength']
            confidence_sum = primary_signal['confidence']
            
            for tf, signals in timeframe_signals.items():
                if tf != primary_tf:
                    for signal in signals:
                        # Same direction signal
                        if (primary_signal['signal_type'] == signal['signal_type'] and 
                            signal['strength'] > self.config['signal_threshold']):
                            confirming_signals += 1
                            signal_strength_sum += signal['strength']
                            confidence_sum += signal['confidence']
            
            # Require minimum confluence
            if confirming_signals >= self.config['confluence_requirement'] - 1:
                # Average strength and confidence
                total_signals = confirming_signals + 1
                avg_strength = signal_strength_sum / total_signals
                avg_confidence = confidence_sum / total_signals
                
                enhanced_signal = EnhancedSignal(
                    symbol=symbol,
                    signal_type=primary_signal['signal_type'],
                    strength=avg_strength,
                    confidence=avg_confidence,
                    timeframe=primary_tf,
                    entry_price=primary_signal['entry_price'],
                    stop_loss=primary_signal['stop_loss'],
                    take_profit=primary_signal['take_profit'],
                    risk_reward_ratio=primary_signal['risk_reward_ratio'],
                    signals_confluence=confirming_signals + 1,
                    market_regime=market_regime,
                    factor_exposure=primary_signal['factor_signals']
                )
                
                confluent_signals.append(enhanced_signal)
        
        return confluent_signals
    
    def _rank_and_filter_signals(self, signals: List[EnhancedSignal], market_regime: str) -> List[EnhancedSignal]:
        """Rank and filter signals by quality."""
        
        if not signals:
            return signals
        
        # Filter by minimum risk/reward ratio
        filtered_signals = [
            signal for signal in signals 
            if signal.risk_reward_ratio >= self.config['min_risk_reward']
        ]
        
        # Score signals
        for signal in filtered_signals:
            # Quality score combines multiple factors
            score = (
                signal.strength * 0.3 +
                signal.confidence * 0.3 +
                min(signal.signals_confluence / 4, 1.0) * 0.2 +
                min(signal.risk_reward_ratio / 4, 1.0) * 0.2
            )
            signal.quality_score = score
        
        # Sort by quality score
        ranked_signals = sorted(filtered_signals, key=lambda x: x.quality_score, reverse=True)
        
        # Return top signals (limit based on market regime)
        max_signals = {
            'bull': 10,
            'bear': 5,
            'sideways': 6,
            'volatile': 3
        }
        
        limit = max_signals.get(market_regime, 6)
        return ranked_signals[:limit] 