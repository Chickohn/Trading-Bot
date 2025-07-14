#!/usr/bin/env python3
"""
Advanced Portfolio Optimization for Maximum Risk-Adjusted Returns
Implements Modern Portfolio Theory, Factor Models, and Dynamic Allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AllocationTarget:
    symbol: str
    target_weight: float
    min_weight: float = 0.0
    max_weight: float = 0.25  # 25% max per position
    sector: str = "Unknown"
    risk_level: str = "Medium"

@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple models:
    1. Modern Portfolio Theory (Markowitz)
    2. Risk Parity
    3. Factor-based optimization
    4. Black-Litterman model
    5. Kelly Criterion for position sizing
    """
    
    def __init__(self, risk_free_rate: float = 0.045):  # 4.5% risk-free rate
        self.risk_free_rate = risk_free_rate
        self.optimization_methods = {
            'max_sharpe': self._optimize_max_sharpe,
            'min_volatility': self._optimize_min_volatility,
            'risk_parity': self._optimize_risk_parity,
            'max_diversification': self._optimize_max_diversification,
            'kelly': self._optimize_kelly_criterion
        }
    
    def optimize_portfolio(
        self,
        returns_data: pd.DataFrame,
        method: str = 'max_sharpe',
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns_data: DataFrame with symbol returns
            method: Optimization method
            constraints: Additional constraints
            target_return: Target return for efficient frontier
        """
        
        if method not in self.optimization_methods:
            raise ValueError(f"Method {method} not supported. Use: {list(self.optimization_methods.keys())}")
        
        # Calculate expected returns and covariance matrix
        expected_returns = self._calculate_expected_returns(returns_data)
        cov_matrix = self._calculate_covariance_matrix(returns_data)
        
        # Apply optimization method
        weights = self.optimization_methods[method](
            expected_returns, cov_matrix, constraints, target_return
        )
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Calculate additional metrics
        max_drawdown = self._calculate_max_drawdown(returns_data, weights)
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        
        # Convert weights to dictionary
        weight_dict = {symbol: weight for symbol, weight in zip(returns_data.columns, weights)}
        
        return OptimizationResult(
            weights=weight_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            diversification_ratio=diversification_ratio
        )
    
    def _calculate_expected_returns(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate expected returns using multiple models."""
        # Ensemble approach: combine historical mean, momentum, and mean reversion
        
        # 1. Historical mean (40% weight)
        historical_mean = returns_data.mean().values
        
        # 2. Momentum signal (30% weight) - recent 21-day returns
        momentum_signal = returns_data.tail(21).mean().values
        
        # 3. Mean reversion signal (30% weight) - inverse of recent performance
        recent_performance = returns_data.tail(63).mean().values  # 3 months
        mean_reversion_signal = -0.5 * recent_performance  # Partial mean reversion
        
        # Combine signals
        expected_returns = (
            0.4 * historical_mean +
            0.3 * momentum_signal +
            0.3 * mean_reversion_signal
        )
        
        return expected_returns
    
    def _calculate_covariance_matrix(self, returns_data: pd.DataFrame) -> np.ndarray:
        """Calculate shrunk covariance matrix using Ledoit-Wolf shrinkage."""
        sample_cov = returns_data.cov().values
        
        # Ledoit-Wolf shrinkage towards single-index model
        n_assets = sample_cov.shape[0]
        target = np.eye(n_assets) * np.trace(sample_cov) / n_assets
        
        # Shrinkage intensity (simplified)
        shrinkage = 0.2  # 20% shrinkage
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        return shrunk_cov
    
    def _optimize_max_sharpe(
        self, 
        expected_returns: np.ndarray, 
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Optimize for maximum Sharpe ratio."""
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds (0% to 25% per asset by default)
        bounds = tuple((0, 0.25) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _optimize_min_volatility(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Optimize for minimum volatility."""
        n_assets = len(expected_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x * expected_returns) - target_return
            })
        
        bounds = tuple((0, 0.25) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _optimize_risk_parity(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Risk parity optimization - equal risk contribution."""
        n_assets = len(expected_returns)
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Minimize sum of squared deviations from equal risk contribution
            target_contrib = portfolio_vol**2 / n_assets
            return np.sum((contrib - target_contrib)**2)
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0.01, 0.25) for _ in range(n_assets))  # Minimum 1% per asset
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _optimize_max_diversification(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Optimize for maximum diversification ratio."""
        n_assets = len(expected_returns)
        asset_vols = np.sqrt(np.diag(cov_matrix))
        
        def negative_diversification_ratio(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            weighted_avg_vol = np.sum(weights * asset_vols)
            return -weighted_avg_vol / portfolio_vol
        
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 0.25) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        return result.x
    
    def _optimize_kelly_criterion(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Optional[Dict] = None,
        target_return: Optional[float] = None
    ) -> np.ndarray:
        """Kelly Criterion for optimal position sizing."""
        # Kelly formula: f = (bp - q) / b where b = odds, p = win probability, q = lose probability
        # For continuous returns: f = μ / σ²
        
        kelly_weights = expected_returns / np.diag(cov_matrix)
        kelly_weights = np.maximum(kelly_weights, 0)  # No short positions
        
        # Normalize to sum to 1
        if kelly_weights.sum() > 0:
            kelly_weights = kelly_weights / kelly_weights.sum()
        else:
            kelly_weights = np.array([1/len(expected_returns)] * len(expected_returns))
        
        # Apply position limits (max 25% per position)
        kelly_weights = np.minimum(kelly_weights, 0.25)
        kelly_weights = kelly_weights / kelly_weights.sum()  # Renormalize
        
        return kelly_weights
    
    def _calculate_max_drawdown(self, returns_data: pd.DataFrame, weights: np.ndarray) -> float:
        """Calculate maximum drawdown of the portfolio."""
        portfolio_returns = (returns_data * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio."""
        asset_vols = np.sqrt(np.diag(cov_matrix))
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        weighted_avg_vol = np.sum(weights * asset_vols)
        return weighted_avg_vol / portfolio_vol
    
    def generate_efficient_frontier(
        self,
        returns_data: pd.DataFrame,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate efficient frontier points."""
        expected_returns = self._calculate_expected_returns(returns_data)
        cov_matrix = self._calculate_covariance_matrix(returns_data)
        
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_vols = []
        
        for target_ret in target_returns:
            try:
                weights = self._optimize_min_volatility(
                    expected_returns, cov_matrix, target_return=target_ret
                )
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                efficient_vols.append(vol)
            except:
                efficient_vols.append(np.nan)
        
        return np.array(target_returns), np.array(efficient_vols)
    
    def calculate_sector_allocation(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate allocation by sector (simplified mapping)."""
        # Simplified sector mapping
        sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Technology', 'NVDA': 'Technology', 'META': 'Technology',
            'TSLA': 'Automotive', 'IONQ': 'Quantum', 'RGTI': 'Quantum',
            'TSM': 'Semiconductors', 'ASML': 'Semiconductors', 'AVGO': 'Semiconductors',
            'AMD': 'Semiconductors', 'ENPH': 'Clean Energy', 'SPX.L': 'International',
            'SAP': 'International', 'SMCI': 'Hardware', 'CRWD': 'Cybersecurity',
            'V': 'Fintech', 'MA': 'Fintech', 'PLTR': 'AI/Data', 'C3AI': 'AI/Data',
            'COIN': 'Crypto', 'MSTR': 'Crypto', 'NEE': 'Utilities'
        }
        
        sector_allocation = {}
        for symbol, weight in weights.items():
            sector = sector_mapping.get(symbol, 'Other')
            sector_allocation[sector] = sector_allocation.get(sector, 0) + weight
        
        return sector_allocation 