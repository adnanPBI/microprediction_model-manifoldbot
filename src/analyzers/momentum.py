"""Market momentum analysis"""

from typing import List, Dict, Optional
import numpy as np
from ..api.models import Market, Bet
from ..api.client import ManifoldClient


class MomentumAnalyzer:
    """Analyzes market price momentum"""

    def __init__(self, client: ManifoldClient, lookback_hours: int = 24):
        """
        Initialize momentum analyzer

        Args:
            client: Manifold API client
            lookback_hours: Hours to look back for momentum
        """
        self.client = client
        self.lookback_hours = lookback_hours

    def analyze(self, market: Market) -> Dict[str, float]:
        """
        Analyze momentum for a market

        Args:
            market: Market to analyze

        Returns:
            Dict with momentum metrics
        """
        try:
            # Get recent bets
            bets = self.client.get_bets(market_id=market.id, limit=100)

            if len(bets) < 2:
                return {
                    "momentum": 0.0,
                    "trend": "flat",
                    "velocity": 0.0,
                    "confidence": 0.0,
                }

            # Filter to lookback period
            recent_bets = self._filter_recent_bets(bets)

            if len(recent_bets) < 2:
                return {
                    "momentum": 0.0,
                    "trend": "flat",
                    "velocity": 0.0,
                    "confidence": 0.0,
                }

            # Calculate momentum
            momentum = self._calculate_momentum(recent_bets)
            velocity = self._calculate_velocity(recent_bets)

            return {
                "momentum": momentum,
                "trend": "bullish" if momentum > 0.01 else "bearish" if momentum < -0.01 else "flat",
                "velocity": velocity,
                "confidence": min(len(recent_bets) / 20.0, 1.0),
            }

        except Exception as e:
            return {
                "momentum": 0.0,
                "trend": "flat",
                "velocity": 0.0,
                "confidence": 0.0,
                "error": str(e),
            }

    def _filter_recent_bets(self, bets: List[Bet]) -> List[Bet]:
        """Filter bets to lookback period"""
        import time
        cutoff = (time.time() - self.lookback_hours * 3600) * 1000
        return [b for b in bets if b.created_time >= cutoff]

    def _calculate_momentum(self, bets: List[Bet]) -> float:
        """
        Calculate price momentum

        Returns:
            Momentum value (positive = upward, negative = downward)
        """
        if len(bets) < 2:
            return 0.0

        # Sort by time
        sorted_bets = sorted(bets, key=lambda b: b.created_time)

        # Calculate price change
        initial_prob = sorted_bets[0].prob_after
        final_prob = sorted_bets[-1].prob_after

        momentum = final_prob - initial_prob
        return momentum

    def _calculate_velocity(self, bets: List[Bet]) -> float:
        """
        Calculate rate of price change

        Returns:
            Velocity (probability change per hour)
        """
        if len(bets) < 2:
            return 0.0

        sorted_bets = sorted(bets, key=lambda b: b.created_time)

        initial_prob = sorted_bets[0].prob_after
        final_prob = sorted_bets[-1].prob_after
        time_diff = (sorted_bets[-1].created_time - sorted_bets[0].created_time) / (1000 * 3600)

        if time_diff == 0:
            return 0.0

        velocity = (final_prob - initial_prob) / time_diff
        return velocity
