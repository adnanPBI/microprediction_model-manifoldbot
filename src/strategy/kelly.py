"""Kelly Criterion betting strategy"""

import numpy as np
from typing import Optional
from ..api.models import Market
from ..predictors.base import Prediction


class KellyBetting:
    """Implements fractional Kelly betting with market impact adjustment"""

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        max_edge: float = 0.4,
        market_impact_adjustment: bool = True,
    ):
        """
        Initialize Kelly betting strategy

        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            min_edge: Minimum edge to bet
            max_edge: Maximum edge to believe (sanity check)
            market_impact_adjustment: Adjust for market impact
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_edge = max_edge
        self.market_impact_adjustment = market_impact_adjustment

    def calculate_bet_size(
        self,
        prediction: Prediction,
        market: Market,
        bankroll: float,
        max_bet: float,
        min_bet: float,
    ) -> Optional[dict]:
        """
        Calculate optimal bet size using Kelly criterion

        Args:
            prediction: Model prediction
            market: Market to bet on
            bankroll: Current bankroll
            max_bet: Maximum bet size
            min_bet: Minimum bet size

        Returns:
            Dict with bet_size and outcome, or None if no bet
        """
        market_prob = market.probability
        predicted_prob = prediction.probability

        # Calculate edge
        edge = predicted_prob - market_prob

        # Check if edge meets minimum threshold
        if abs(edge) < self.min_edge:
            return None

        # Sanity check on edge
        if abs(edge) > self.max_edge:
            return None

        # Determine outcome (YES or NO)
        outcome = "YES" if edge > 0 else "NO"
        effective_pred_prob = predicted_prob if edge > 0 else (1 - predicted_prob)
        effective_market_prob = market_prob if edge > 0 else (1 - market_prob)

        # [FIX] Clamp probability to avoid division by zero or infinity
        # Manifold markets can hit 0.0 or 1.0 occasionally
        effective_market_prob = max(0.001, min(0.999, effective_market_prob))

        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = 1 - p
        # For binary markets: f = p - (1-p) / odds
        odds = (1 - effective_market_prob) / effective_market_prob

        # [FIX] Double check odds isn't zero (redundant but safe)
        if odds <= 0:
            return None

        kelly_fraction_full = (effective_pred_prob * (odds + 1) - 1) / odds

        # Apply fractional Kelly
        kelly_fraction_adj = kelly_fraction_full * self.kelly_fraction

        # Adjust for confidence
        kelly_fraction_adj *= prediction.confidence

        # Calculate bet size
        bet_size = bankroll * kelly_fraction_adj

        # Apply market impact adjustment
        if self.market_impact_adjustment:
            bet_size = self._adjust_for_market_impact(bet_size, market)

        # Apply limits
        bet_size = max(min_bet, min(bet_size, max_bet))

        # Don't bet if size is too small
        if bet_size < min_bet:
            return None

        return {
            "outcome": outcome,
            "bet_size": bet_size,
            "edge": edge,
            "kelly_fraction": kelly_fraction_adj,
            "predicted_prob": predicted_prob,
            "market_prob": market_prob,
        }

    def _adjust_for_market_impact(self, bet_size: float, market: Market) -> float:
        """
        Adjust bet size based on market liquidity to reduce impact

        Args:
            bet_size: Proposed bet size
            market: Market information

        Returns:
            Adjusted bet size
        """
        liquidity = market.liquidity

        if liquidity == 0:
            return bet_size * 0.1  # Very conservative if no liquidity info

        # Limit bet to some fraction of liquidity to avoid excessive slippage
        max_fraction = 0.1  # Max 10% of liquidity
        max_bet_from_liquidity = liquidity * max_fraction

        return min(bet_size, max_bet_from_liquidity)
