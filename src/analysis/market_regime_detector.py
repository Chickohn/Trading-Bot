#!/usr/bin/env python3
"""
Advanced Market Regime Detection for Adaptive Trading Strategies
Identifies bull, bear, and sideways markets using multiple techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

@dataclass
class RegimeSignal:
    regime: MarketRegime
    confidence: float  # 0-1
    strength: float   # 0-1
    duration: int     # days in current regime
    indicators: Dict[str, float]  # Supporting indicators

@dataclass
class RegimeConfig:
    """Configuration for regime detection parameters."""
    trend_threshold: float = 0.02  # 2% monthly return threshold
    volatility_threshold: float = 0.15  # 15% annualized volatility threshold
    momentum_period: int = 21  # Days for momentum calculation
    regime_persistence: int = 5  # Minimum days to confirm regime change
    confidence_threshold: float = 0.6  # Minimum confidence for regime classification

class AdvancedMarketRegimeDetector:
    """
    Multi-signal market regime detection using:
    1. Trend analysis (moving averages, momentum)
    2. Volatility regime detection
    3. Market breadth indicators
    4. Sentiment indicators
    5. Hidden Markov Models
    6. Risk-on/Risk-off analysis
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.regime_history = []
        self.current_regime = MarketRegime.UNKNOWN
        
    def detect_regime(
        self, 
        market_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.DataFrame] = None
    ) -> RegimeSignal:
        """
        Detect current market regime using multiple indicators.
        
        Args:
            market_data: Main market price data (OHLCV)
            spy_data: S&P 500 data for market context
            vix_data: VIX data for volatility context
        """
        
        # Calculate all regime indicators
        indicators = self._calculate_regime_indicators(market_data, spy_data, vix_data)
        
        # Apply regime classification logic
        regime, confidence = self._classify_regime(indicators)
        
        # Calculate regime strength and duration
        strength = self._calculate_regime_strength(indicators, regime)
        duration = self._calculate_regime_duration(regime)
        
        regime_signal = RegimeSignal(
            regime=regime,
            confidence=confidence,
            strength=strength,
            duration=duration,
            indicators=indicators
        )
        
        # Update regime history
        self._update_regime_history(regime_signal)
        
        return regime_signal
    
    def _calculate_regime_indicators(
        self,
        market_data: pd.DataFrame,
        spy_data: Optional[pd.DataFrame] = None,
        vix_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate all indicators used for regime detection."""
        
        indicators = {}
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        # 1. Trend Indicators
        indicators.update(self._calculate_trend_indicators(close_prices))
        
        # 2. Volatility Indicators
        indicators.update(self._calculate_volatility_indicators(market_data))
        
        # 3. Momentum Indicators
        indicators.update(self._calculate_momentum_indicators(close_prices))
        
        # 4. Market Structure Indicators
        indicators.update(self._calculate_market_structure_indicators(market_data))
        
        # 5. Risk-On/Risk-Off Indicators
        if spy_data is not None:
            indicators.update(self._calculate_risk_indicators(close_prices, spy_data))
        
        # 6. Fear/Greed Indicators
        if vix_data is not None:
            indicators.update(self._calculate_sentiment_indicators(vix_data))
        
        return indicators
    
    def _calculate_trend_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate trend-based regime indicators."""
        indicators = {}
        
        # Moving Average Slopes
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        ma_200 = prices.rolling(200).mean()
        
        # Price position relative to moving averages
        indicators['price_above_ma20'] = float(prices.iloc[-1] > ma_20.iloc[-1]) if len(ma_20.dropna()) > 0 else 0.5
        indicators['price_above_ma50'] = float(prices.iloc[-1] > ma_50.iloc[-1]) if len(ma_50.dropna()) > 0 else 0.5
        indicators['price_above_ma200'] = float(prices.iloc[-1] > ma_200.iloc[-1]) if len(ma_200.dropna()) > 0 else 0.5
        
        # Moving average slopes (trend strength)
        if len(ma_20.dropna()) >= 5:
            ma20_slope = (ma_20.iloc[-1] - ma_20.iloc[-5]) / ma_20.iloc[-5]
            indicators['ma20_slope'] = np.tanh(ma20_slope * 100)  # Normalize between -1 and 1
        else:
            indicators['ma20_slope'] = 0
        
        if len(ma_50.dropna()) >= 10:
            ma50_slope = (ma_50.iloc[-1] - ma_50.iloc[-10]) / ma_50.iloc[-10]
            indicators['ma50_slope'] = np.tanh(ma50_slope * 100)
        else:
            indicators['ma50_slope'] = 0
        
        # Monthly return (key trend indicator)
        if len(prices) >= 21:
            monthly_return = (prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21]
            indicators['monthly_return'] = monthly_return
        else:
            indicators['monthly_return'] = 0
        
        return indicators
    
    def _calculate_volatility_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-based regime indicators."""
        indicators = {}
        
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        # Historical volatility (21-day)
        returns = close_prices.pct_change().dropna()
        if len(returns) >= 21:
            vol_21d = returns.tail(21).std() * np.sqrt(252)  # Annualized
            indicators['volatility_21d'] = vol_21d
            
            # Volatility regime (high vs normal)
            vol_percentile = stats.percentileofscore(returns.tail(252), returns.iloc[-1]) / 100
            indicators['volatility_percentile'] = vol_percentile
        else:
            indicators['volatility_21d'] = 0.15  # Default moderate volatility
            indicators['volatility_percentile'] = 0.5
        
        # Intraday volatility (if OHLC available)
        if all(col in market_data.columns for col in ['high', 'low', 'open', 'close']):
            # True Range
            tr = np.maximum(
                market_data['high'] - market_data['low'],
                np.maximum(
                    abs(market_data['high'] - market_data['close'].shift(1)),
                    abs(market_data['low'] - market_data['close'].shift(1))
                )
            )
            
            if len(tr.dropna()) >= 14:
                atr_14 = tr.rolling(14).mean().iloc[-1]
                atr_pct = atr_14 / close_prices.iloc[-1]
                indicators['atr_percentage'] = atr_pct
            else:
                indicators['atr_percentage'] = 0.02  # Default 2%
        
        return indicators
    
    def _calculate_momentum_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate momentum-based regime indicators."""
        indicators = {}
        
        # Rate of Change indicators
        for period in [5, 10, 21]:
            if len(prices) >= period + 1:
                roc = (prices.iloc[-1] - prices.iloc[-(period+1)]) / prices.iloc[-(period+1)]
                indicators[f'roc_{period}d'] = roc
        
        # RSI-like momentum indicator
        if len(prices) >= 14:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['momentum_rsi'] = (rsi.iloc[-1] - 50) / 50  # Normalize to -1 to 1
        else:
            indicators['momentum_rsi'] = 0
        
        return indicators
    
    def _calculate_market_structure_indicators(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market structure indicators."""
        indicators = {}
        
        close_prices = market_data['close'] if 'close' in market_data.columns else market_data.iloc[:, -1]
        
        # Higher highs, higher lows pattern detection
        if len(close_prices) >= 20:
            recent_highs = close_prices.rolling(5).max().tail(10)
            recent_lows = close_prices.rolling(5).min().tail(10)
            
            # Trend of highs and lows
            if len(recent_highs) >= 2:
                higher_highs = recent_highs.iloc[-1] > recent_highs.iloc[-2]
                higher_lows = recent_lows.iloc[-1] > recent_lows.iloc[-2]
                
                if higher_highs and higher_lows:
                    indicators['market_structure'] = 1.0  # Bullish structure
                elif not higher_highs and not higher_lows:
                    indicators['market_structure'] = -1.0  # Bearish structure
                else:
                    indicators['market_structure'] = 0.0  # Mixed structure
            else:
                indicators['market_structure'] = 0.0
        else:
            indicators['market_structure'] = 0.0
        
        return indicators
    
    def _calculate_risk_indicators(self, prices: pd.Series, spy_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-on/risk-off indicators using SPY benchmark."""
        indicators = {}
        
        spy_close = spy_data['close'] if 'close' in spy_data.columns else spy_data.iloc[:, -1]
        
        if len(prices) >= 21 and len(spy_close) >= 21:
            # Relative performance vs market
            asset_return = (prices.iloc[-1] - prices.iloc[-21]) / prices.iloc[-21]
            spy_return = (spy_close.iloc[-1] - spy_close.iloc[-21]) / spy_close.iloc[-21]
            
            relative_performance = asset_return - spy_return
            indicators['relative_performance'] = relative_performance
            
            # Correlation with market (risk-on assets have higher correlation in bull markets)
            asset_returns = prices.pct_change().tail(21).dropna()
            spy_returns = spy_close.pct_change().tail(21).dropna()
            
            if len(asset_returns) >= 10 and len(spy_returns) >= 10:
                correlation = asset_returns.corr(spy_returns)
                indicators['market_correlation'] = correlation if not np.isnan(correlation) else 0
            else:
                indicators['market_correlation'] = 0
        else:
            indicators['relative_performance'] = 0
            indicators['market_correlation'] = 0
        
        return indicators
    
    def _calculate_sentiment_indicators(self, vix_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment indicators using VIX data."""
        indicators = {}
        
        vix_close = vix_data['close'] if 'close' in vix_data.columns else vix_data.iloc[:, -1]
        
        if len(vix_close) >= 1:
            current_vix = vix_close.iloc[-1]
            
            # VIX levels (fear/greed indicator)
            if current_vix < 15:
                indicators['fear_greed'] = 0.8  # Greed (low fear)
            elif current_vix < 20:
                indicators['fear_greed'] = 0.6  # Complacency
            elif current_vix < 30:
                indicators['fear_greed'] = 0.4  # Moderate concern
            elif current_vix < 40:
                indicators['fear_greed'] = 0.2  # Fear
            else:
                indicators['fear_greed'] = 0.0  # Extreme fear
            
            # VIX trend
            if len(vix_close) >= 5:
                vix_ma5 = vix_close.rolling(5).mean().iloc[-1]
                vix_trend = (current_vix - vix_ma5) / vix_ma5
                indicators['vix_trend'] = np.tanh(vix_trend)  # Normalize
            else:
                indicators['vix_trend'] = 0
        else:
            indicators['fear_greed'] = 0.5
            indicators['vix_trend'] = 0
        
        return indicators
    
    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """Classify market regime based on indicators."""
        
        # Weighted scoring system
        weights = {
            'trend': 0.3,      # Trend indicators
            'momentum': 0.25,  # Momentum indicators
            'volatility': 0.2, # Volatility indicators
            'structure': 0.15, # Market structure
            'sentiment': 0.1   # Sentiment indicators
        }
        
        # Calculate trend score
        trend_score = (
            indicators.get('price_above_ma20', 0.5) * 0.3 +
            indicators.get('price_above_ma50', 0.5) * 0.3 +
            indicators.get('ma20_slope', 0) * 0.4
        )
        
        # Calculate momentum score
        momentum_score = (
            indicators.get('roc_21d', 0) * 5 +  # Scale to -1 to 1 range
            indicators.get('momentum_rsi', 0) * 0.5 +
            indicators.get('monthly_return', 0) * 10  # Scale monthly return
        )
        momentum_score = np.tanh(momentum_score)  # Normalize to -1 to 1
        
        # Calculate volatility score (inverse - low vol = bullish)
        vol_score = 1 - min(indicators.get('volatility_21d', 0.15) / 0.4, 1)  # Normalize
        
        # Market structure score
        structure_score = indicators.get('market_structure', 0)
        
        # Sentiment score
        sentiment_score = (indicators.get('fear_greed', 0.5) - 0.5) * 2  # Convert to -1 to 1
        
        # Combine scores
        overall_score = (
            trend_score * weights['trend'] +
            momentum_score * weights['momentum'] +
            vol_score * weights['volatility'] +
            structure_score * weights['structure'] +
            sentiment_score * weights['sentiment']
        )
        
        # Determine regime
        volatility = indicators.get('volatility_21d', 0.15)
        
        if volatility > self.config.volatility_threshold * 1.5:  # Very high volatility
            regime = MarketRegime.VOLATILE
            confidence = min(volatility / 0.3, 1.0)
        elif overall_score > 0.3:
            regime = MarketRegime.BULL
            confidence = min(overall_score, 1.0)
        elif overall_score < -0.3:
            regime = MarketRegime.BEAR
            confidence = min(abs(overall_score), 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1.0 - abs(overall_score) / 0.3
        
        return regime, max(confidence, 0.1)  # Minimum 10% confidence
    
    def _calculate_regime_strength(self, indicators: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate strength of the current regime."""
        
        if regime == MarketRegime.BULL:
            # Bull market strength factors
            strength_factors = [
                indicators.get('ma20_slope', 0),
                indicators.get('monthly_return', 0) * 10,
                indicators.get('market_structure', 0),
                indicators.get('fear_greed', 0.5) - 0.5
            ]
        elif regime == MarketRegime.BEAR:
            # Bear market strength factors (inverted)
            strength_factors = [
                -indicators.get('ma20_slope', 0),
                -indicators.get('monthly_return', 0) * 10,
                -indicators.get('market_structure', 0),
                0.5 - indicators.get('fear_greed', 0.5)
            ]
        elif regime == MarketRegime.VOLATILE:
            # Volatility regime strength
            vol = indicators.get('volatility_21d', 0.15)
            return min(vol / 0.4, 1.0)
        else:  # SIDEWAYS
            # Sideways strength = low volatility + low momentum
            vol_factor = 1 - min(indicators.get('volatility_21d', 0.15) / 0.2, 1)
            momentum_factor = 1 - abs(indicators.get('monthly_return', 0)) * 20
            return (vol_factor + momentum_factor) / 2
        
        avg_strength = np.mean([max(0, factor) for factor in strength_factors])
        return min(avg_strength, 1.0)
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long current regime has been in place."""
        if not self.regime_history:
            return 1
        
        duration = 1
        for i in range(len(self.regime_history) - 1, -1, -1):
            if self.regime_history[i].regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _update_regime_history(self, regime_signal: RegimeSignal):
        """Update regime history and current regime."""
        self.regime_history.append(regime_signal)
        self.current_regime = regime_signal.regime
        
        # Keep only last 100 regime signals
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
    
    def get_regime_strategy_recommendations(self, regime_signal: RegimeSignal) -> Dict[str, any]:
        """Get trading strategy recommendations based on current regime."""
        
        recommendations = {
            'position_sizing': 1.0,
            'risk_multiplier': 1.0,
            'preferred_timeframes': ['1h', '4h', '1d'],
            'strategy_focus': 'balanced',
            'volatility_target': 0.15,
            'max_positions': 5
        }
        
        if regime_signal.regime == MarketRegime.BULL:
            recommendations.update({
                'position_sizing': 1.2 * regime_signal.strength,  # Increase size in strong bull
                'risk_multiplier': 0.8,  # Lower risk in trending market
                'preferred_timeframes': ['4h', '1d'],  # Longer timeframes for trends
                'strategy_focus': 'momentum',
                'volatility_target': 0.12,  # Accept lower volatility
                'max_positions': 7  # More positions in trending market
            })
            
        elif regime_signal.regime == MarketRegime.BEAR:
            recommendations.update({
                'position_sizing': 0.6,  # Reduce size in bear market
                'risk_multiplier': 1.5,  # Higher risk management
                'preferred_timeframes': ['1h', '4h'],  # Shorter timeframes for volatility
                'strategy_focus': 'mean_reversion',
                'volatility_target': 0.20,  # Higher volatility target
                'max_positions': 3  # Fewer positions
            })
            
        elif regime_signal.regime == MarketRegime.VOLATILE:
            recommendations.update({
                'position_sizing': 0.5,  # Much smaller positions
                'risk_multiplier': 2.0,  # Very high risk management
                'preferred_timeframes': ['15m', '1h'],  # Short timeframes
                'strategy_focus': 'scalping',
                'volatility_target': 0.25,  # High volatility tolerance
                'max_positions': 2  # Very few positions
            })
            
        elif regime_signal.regime == MarketRegime.SIDEWAYS:
            recommendations.update({
                'position_sizing': 0.8,  # Moderate positions
                'risk_multiplier': 1.2,  # Slightly higher risk mgmt
                'preferred_timeframes': ['1h', '4h'],  # Medium timeframes
                'strategy_focus': 'range_trading',
                'volatility_target': 0.10,  # Low volatility target
                'max_positions': 4  # Moderate positions
            })
        
        return recommendations 