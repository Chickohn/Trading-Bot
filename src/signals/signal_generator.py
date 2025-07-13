"""
Signal generation system for trading decisions.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import structlog

from models.base_model import PredictionResult, SignalType, ModelEnsemble
from data.base import OHLCV, MarketData
from utils.config import config, trading, risk

logger = structlog.get_logger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class TradingSignal:
    """Complete trading signal with all metadata."""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    timestamp: datetime
    price: float
    volume: float
    models_used: List[str]
    model_predictions: Dict[str, PredictionResult]
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for storage/logging."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'volume': self.volume,
            'models_used': self.models_used,
            'metadata': self.metadata or {}
        }


class SignalGenerator:
    """Main signal generation system."""
    
    def __init__(self, models: List[Any], config_section=None):
        self.models = models
        self.config = config_section or trading
        self.ensemble = ModelEnsemble(models) if len(models) > 1 else None
        self.signal_history: List[TradingSignal] = []
        self.last_signals: Dict[str, TradingSignal] = {}
        self.signal_cooldowns: Dict[str, datetime] = {}
        
    async def generate_signal(
        self,
        symbol: str,
        market_data: MarketData,
        historical_data: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol."""
        try:
            # Check cooldown period
            if self._is_in_cooldown(symbol):
                logger.debug(f"Signal for {symbol} is in cooldown period")
                return None
            
            # Get predictions from all models
            predictions = await self._get_model_predictions(symbol, historical_data)
            
            if not predictions:
                logger.warning(f"No predictions available for {symbol}")
                return None
            
            # Combine predictions into final signal
            signal = self._combine_predictions(symbol, predictions, market_data)
            
            if signal:
                # Store signal
                self.signal_history.append(signal)
                self.last_signals[symbol] = signal
                self.signal_cooldowns[symbol] = datetime.now()
                
                logger.info(f"Generated {signal.signal_type.value} signal for {symbol} "
                          f"with confidence {signal.confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _get_model_predictions(
        self,
        symbol: str,
        historical_data: pd.DataFrame
    ) -> Dict[str, PredictionResult]:
        """Get predictions from all available models."""
        predictions = {}
        
        for model in self.models:
            try:
                if model.is_trained:
                    prediction = model.predict(historical_data)
                    predictions[model.config.model_name] = prediction
                else:
                    logger.warning(f"Model {model.config.model_name} is not trained")
            except Exception as e:
                logger.error(f"Error getting prediction from {model.config.model_name}: {e}")
        
        return predictions
    
    def _combine_predictions(
        self,
        symbol: str,
        predictions: Dict[str, PredictionResult],
        market_data: MarketData
    ) -> Optional[TradingSignal]:
        """Combine multiple model predictions into a single signal."""
        if not predictions:
            return None
        
        # Use ensemble if available
        if self.ensemble and len(predictions) > 1:
            # Create a temporary DataFrame for ensemble prediction
            # This is a simplified approach - in practice you'd want to pass the actual data
            ensemble_prediction = self.ensemble.predict(pd.DataFrame())
            final_signal = ensemble_prediction.signal
            final_confidence = ensemble_prediction.confidence
        else:
            # Use the best single model prediction
            best_prediction = max(predictions.values(), key=lambda p: p.confidence)
            final_signal = best_prediction.signal
            final_confidence = best_prediction.confidence
        
        # Apply signal filters
        if not self._passes_signal_filters(symbol, final_signal, final_confidence, market_data):
            return None
        
        # Determine signal strength
        strength = self._determine_signal_strength(final_confidence, predictions)
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_type=final_signal,
            strength=strength,
            confidence=final_confidence,
            timestamp=datetime.now(),
            price=market_data.ohlcv.close,
            volume=market_data.ohlcv.volume,
            models_used=list(predictions.keys()),
            model_predictions=predictions,
            metadata={
                'individual_predictions': {
                    name: pred.to_dict() for name, pred in predictions.items()
                }
            }
        )
        
        return signal
    
    def _passes_signal_filters(
        self,
        symbol: str,
        signal: SignalType,
        confidence: float,
        market_data: MarketData
    ) -> bool:
        """Check if signal passes all filters."""
        # Confidence threshold
        if confidence < self.config.pattern_threshold:
            return False
        
        # Volume filter
        if market_data.ohlcv.volume < 1000:  # Minimum volume
            return False
        
        # Price filter (avoid penny stocks)
        if market_data.ohlcv.close < 1.0:
            return False
        
        # Volatility filter
        if hasattr(market_data, 'atr') and market_data.atr:
            if market_data.atr / market_data.ohlcv.close < 0.001:  # Too low volatility
                return False
        
        return True
    
    def _determine_signal_strength(
        self,
        confidence: float,
        predictions: Dict[str, PredictionResult]
    ) -> SignalStrength:
        """Determine signal strength based on confidence and model agreement."""
        # Base strength on confidence
        if confidence >= 0.9:
            base_strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            base_strength = SignalStrength.STRONG
        elif confidence >= 0.7:
            base_strength = SignalStrength.MEDIUM
        else:
            base_strength = SignalStrength.WEAK
        
        # Adjust based on model agreement
        if len(predictions) > 1:
            signals = [pred.signal for pred in predictions.values()]
            agreement = len(set(signals)) == 1  # All models agree
            
            if agreement and base_strength in [SignalStrength.WEAK, SignalStrength.MEDIUM]:
                return SignalStrength.STRONG
            elif not agreement and base_strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                return SignalStrength.MEDIUM
        
        return base_strength
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.signal_cooldowns:
            return False
        
        last_signal_time = self.signal_cooldowns[symbol]
        cooldown_duration = timedelta(seconds=self.config.signal_cooldown)
        
        return datetime.now() - last_signal_time < cooldown_duration
    
    def get_signal_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TradingSignal]:
        """Get signal history with optional filters."""
        history = self.signal_history
        
        if symbol:
            history = [s for s in history if s.symbol == symbol]
        
        if start_time:
            history = [s for s in history if s.timestamp >= start_time]
        
        if end_time:
            history = [s for s in history if s.timestamp <= end_time]
        
        return history
    
    def get_signal_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about generated signals."""
        history = self.get_signal_history(symbol)
        
        if not history:
            return {}
        
        # Count signals by type
        signal_counts = {}
        for signal_type in SignalType:
            signal_counts[signal_type.value] = len([
                s for s in history if s.signal_type == signal_type
            ])
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in history])
        
        # Calculate success rate (simplified - would need actual trade results)
        recent_signals = [s for s in history if s.timestamp > datetime.now() - timedelta(days=7)]
        
        return {
            'total_signals': len(history),
            'signal_counts': signal_counts,
            'average_confidence': avg_confidence,
            'recent_signals': len(recent_signals),
            'last_signal_time': max(s.timestamp for s in history).isoformat() if history else None
        }
    
    def clear_history(self):
        """Clear signal history."""
        self.signal_history.clear()
        self.last_signals.clear()
        self.signal_cooldowns.clear()
        logger.info("Signal history cleared")


class SignalValidator:
    """Validate and filter trading signals."""
    
    def __init__(self, config_section=None):
        self.config = config_section or risk
    
    def validate_signal(self, signal: TradingSignal, market_data: MarketData) -> bool:
        """Validate a trading signal."""
        # Basic validation
        if not signal or not market_data:
            return False
        
        # Price validation
        if signal.price <= 0:
            return False
        
        # Volume validation
        if signal.volume <= 0:
            return False
        
        # Confidence validation
        if signal.confidence < 0 or signal.confidence > 1:
            return False
        
        # Signal strength validation
        if signal.strength == SignalStrength.WEAK and signal.confidence < 0.6:
            return False
        
        return True 