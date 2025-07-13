"""
Technical indicators for feature engineering and pattern detection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import structlog

# Try to import talib, fall back to ta library if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = structlog.get_logger(__name__)
    logger.warning("TA-Lib not available, using ta library instead")

logger = structlog.get_logger(__name__)


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    lookback_period: int = 14
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    upper_band: float = 0.8
    lower_band: float = 0.2
    std_dev: float = 2.0


class TechnicalIndicators:
    """Comprehensive technical indicators calculator."""
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame."""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Make a copy to avoid modifying original
        result_df = df.copy()
        
        # Trend Indicators
        result_df = self._add_trend_indicators(result_df)
        
        # Momentum Indicators
        result_df = self._add_momentum_indicators(result_df)
        
        # Volatility Indicators
        result_df = self._add_volatility_indicators(result_df)
        
        # Volume Indicators
        result_df = self._add_volume_indicators(result_df)
        
        # Pattern Recognition
        result_df = self._add_pattern_indicators(result_df)
        
        # Oscillators
        result_df = self._add_oscillator_indicators(result_df)
        
        return result_df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        try:
            if TALIB_AVAILABLE:
                # Moving Averages
                df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
                df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
                df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
                df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
                df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
                
                # Exponential Moving Averages
                df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
                df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(
                    df['close'],
                    fastperiod=self.config.fast_period,
                    slowperiod=self.config.slow_period,
                    signalperiod=self.config.signal_period
                )
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist
                
                # Parabolic SAR
                df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
                
                # ADX (Average Directional Index)
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=self.config.lookback_period)
                
                # Aroon Oscillator
                aroon_down, aroon_up = talib.AROON(df['high'], df['low'], timeperiod=self.config.lookback_period)
                df['aroon_down'] = aroon_down
                df['aroon_up'] = aroon_up
                df['aroon_osc'] = aroon_up - aroon_down
            else:
                # Fallback to ta library
                import ta
                df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
                df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
                df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
                df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
                
                df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
                df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
                df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
                
                # MACD using ta library
                macd = ta.trend.MACD(df['close'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_histogram'] = macd.macd_diff()
                
                # ADX
                df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=self.config.lookback_period)
                
                # Aroon
                aroon = ta.trend.AroonIndicator(df['high'], df['low'], window=self.config.lookback_period)
                df['aroon_down'] = aroon.aroon_down()
                df['aroon_up'] = aroon.aroon_up()
                df['aroon_osc'] = aroon.aroon_indicator()
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        try:
            if TALIB_AVAILABLE:
                # RSI
                df['rsi'] = talib.RSI(df['close'], timeperiod=self.config.lookback_period)
                
                # Stochastic Oscillator
                slowk, slowd = talib.STOCH(
                    df['high'], df['low'], df['close'],
                    fastk_period=14, slowk_period=3, slowd_period=3
                )
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd
                
                # Williams %R
                df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=self.config.lookback_period)
                
                # CCI (Commodity Channel Index)
                df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=self.config.lookback_period)
                
                # ROC (Rate of Change)
                df['roc'] = talib.ROC(df['close'], timeperiod=10)
                
                # Momentum
                df['momentum'] = talib.MOM(df['close'], timeperiod=10)
            else:
                # Fallback to ta library
                import ta
                df['rsi'] = ta.momentum.rsi(df['close'], window=self.config.lookback_period)
                
                # Stochastic
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
                df['stoch_k'] = stoch.stoch()
                df['stoch_d'] = stoch.stoch_signal()
                
                # Williams %R
                df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=self.config.lookback_period)
                
                # CCI
                df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=self.config.lookback_period)
                
                # ROC
                df['roc'] = ta.momentum.roc(df['close'], window=10)
                
                # Momentum
                df['momentum'] = ta.momentum.roc(df['close'], window=10)  # Using ROC as momentum proxy
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        try:
            if TALIB_AVAILABLE:
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(
                    df['close'],
                    timeperiod=20,
                    nbdevup=self.config.std_dev,
                    nbdevdn=self.config.std_dev,
                    matype=0
                )
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
                df['bb_width'] = (upper - lower) / middle
                df['bb_position'] = (df['close'] - lower) / (upper - lower)
                
                # Average True Range
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.config.lookback_period)
                
                # Standard Deviation
                df['std_dev'] = talib.STDDEV(df['close'], timeperiod=20, nbdev=1)
            else:
                # Fallback to ta library
                import ta
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df['close'])
                df['bb_upper'] = bb.bollinger_hband()
                df['bb_middle'] = bb.bollinger_mavg()
                df['bb_lower'] = bb.bollinger_lband()
                df['bb_width'] = bb.bollinger_wband()
                df['bb_position'] = bb.bollinger_pband()
                
                # Average True Range
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.config.lookback_period)
                
                # Standard Deviation
                df['std_dev'] = df['close'].rolling(window=20).std()
            
            # Keltner Channels (common calculation)
            df['keltner_upper'] = df['ema_20'] + (df['atr'] * 2)
            df['keltner_lower'] = df['ema_20'] - (df['atr'] * 2)
            
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        try:
            if TALIB_AVAILABLE:
                # On Balance Volume
                df['obv'] = talib.OBV(df['close'], df['volume'])
                
                # Volume SMA
                df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # Chaikin Money Flow
                df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
                
                # Money Flow Index
                df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=self.config.lookback_period)
                
                # Accumulation/Distribution Line
                df['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            else:
                # Fallback to ta library
                import ta
                # On Balance Volume
                df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
                
                # Volume SMA
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
                
                # Chaikin Money Flow
                df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], window=20)
                
                # Money Flow Index
                df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=self.config.lookback_period)
                
                # Accumulation/Distribution Line
                df['ad'] = ta.volume.acc_dist_index(df['high'], df['low'], df['close'], df['volume'])
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
        
        return df
    
    def _add_pattern_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition."""
        try:
            if TALIB_AVAILABLE:
                # Bullish Patterns
                df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
                df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
                df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
                df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
                
                # Bearish Patterns
                df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
                df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
                df['three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
                
                # Neutral Patterns
                df['spinning_top'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
                df['high_wave'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
            else:
                # Basic pattern detection without TA-Lib
                logger.info("TA-Lib not available, skipping advanced candlestick patterns")
                # You can implement basic pattern detection here if needed
                pass
            
        except Exception as e:
            logger.error(f"Error calculating pattern indicators: {e}")
        
        return df
    
    def _add_oscillator_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add oscillator indicators."""
        try:
            if TALIB_AVAILABLE:
                # Ultimate Oscillator
                df['ult_osc'] = talib.ULTOSC(df['high'], df['low'], df['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
                
                # TRIX
                df['trix'] = talib.TRIX(df['close'], timeperiod=30)
                
                # Detrended Price Oscillator
                df['dpo'] = talib.DPO(df['close'], timeperiod=20)
                
                # Percentage Price Oscillator
                df['ppo'] = talib.PPO(df['close'], fastperiod=12, slowperiod=26, matype=0)
            else:
                # Fallback to ta library or basic calculations
                import ta
                # TRIX
                df['trix'] = ta.trend.trix(df['close'], window=30)
                
                # Percentage Price Oscillator
                df['ppo'] = ta.momentum.percentage_price_oscillator(df['close'])
                
                # Skip Ultimate Oscillator and DPO as they're not available in ta library
                logger.info("Some oscillators not available without TA-Lib")
            
        except Exception as e:
            logger.error(f"Error calculating oscillator indicators: {e}")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns generated by indicators."""
        return [
            # Trend indicators
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'sar', 'adx', 'aroon_down', 'aroon_up', 'aroon_osc',
            
            # Momentum indicators
            'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'roc', 'momentum',
            
            # Volatility indicators
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr', 'std_dev', 'keltner_upper', 'keltner_lower',
            
            # Volume indicators
            'obv', 'volume_sma', 'volume_ratio', 'cmf', 'mfi', 'ad',
            
            # Pattern indicators
            'doji', 'hammer', 'engulfing', 'morning_star', 'three_white_soldiers',
            'shooting_star', 'hanging_man', 'evening_star', 'three_black_crows',
            'spinning_top', 'high_wave',
            
            # Oscillator indicators
            'ult_osc', 'trix', 'dpo', 'ppo'
        ]
    
    def get_signal_features(self) -> List[str]:
        """Get list of features commonly used for signal generation."""
        return [
            'rsi', 'macd', 'macd_signal', 'bb_position', 'stoch_k', 'stoch_d',
            'adx', 'cci', 'mfi', 'volume_ratio', 'atr', 'std_dev'
        ]
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML models."""
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining NaN values
        df = df.fillna(method='bfill')
        
        # Drop rows with any remaining NaN values
        df = df.dropna()
        
        return df 