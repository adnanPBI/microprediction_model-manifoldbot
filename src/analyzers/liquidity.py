"""Market liquidity analysis"""

from typing import Dict
from ..api.models import Market


class LiquidityAnalyzer:
    """Analyzes market liquidity"""

    def __init__(self, min_liquidity: float = 100):
        """
        Initialize liquidity analyzer

        Args:
            min_liquidity: Minimum acceptable liquidity
        """
        self.min_liquidity = min_liquidity

    def analyze(self, market: Market) -> Dict[str, any]:
        """
        Analyze liquidity for a market

        Args:
            market: Market to analyze

        Returns:
            Dict with liquidity metrics
        """
        liquidity = market.liquidity

        # Calculate liquidity score (0-1)
        if liquidity >= self.min_liquidity * 10:
            score = 1.0
        elif liquidity >= self.min_liquidity:
            score = 0.5 + 0.5 * (liquidity - self.min_liquidity) / (self.min_liquidity * 9)
        else:
            score = 0.5 * liquidity / self.min_liquidity

        return {
            "liquidity": liquidity,
            "liquidity_score": score,
            "is_liquid": liquidity >= self.min_liquidity,
            "volume_24h": market.volume_24_hours,
            "total_volume": market.volume,
        }
