"""Risk management for betting"""

from typing import Dict, List, Optional
from ..api.models import Market, Bet


class RiskManager:
    """Manages portfolio risk and position sizing"""

    def __init__(
        self,
        max_portfolio_exposure: float = 0.5,
        max_position_size: float = 0.1,
        min_liquidity: float = 100,
        stop_loss: float = -0.2,
    ):
        """
        Initialize risk manager

        Args:
            max_portfolio_exposure: Max fraction of bankroll to risk
            max_position_size: Max fraction of bankroll per position
            min_liquidity: Minimum market liquidity required
            stop_loss: Stop loss threshold (e.g., -0.2 = exit at 20% loss)
        """
        self.max_portfolio_exposure = max_portfolio_exposure
        self.max_position_size = max_position_size
        self.min_liquidity = min_liquidity
        self.stop_loss = stop_loss

    def check_market_eligibility(self, market: Market) -> tuple[bool, str]:
        """
        Check if market is eligible for betting

        Args:
            market: Market to check

        Returns:
            (is_eligible, reason)
        """
        # Check if market is resolved
        if market.is_resolved:
            return False, "Market is already resolved"

        # Check if market is binary
        if not market.is_binary:
            return False, "Only binary markets supported"

        # Check liquidity
        if market.liquidity < self.min_liquidity:
            return False, f"Liquidity too low: {market.liquidity:.2f} < {self.min_liquidity}"

        # Check if market is closing soon
        if market.time_until_close is not None and market.time_until_close < 1:
            return False, "Market closing too soon"

        return True, "Market eligible"

    def check_bet_eligibility(
        self,
        bet_size: float,
        bankroll: float,
        current_positions: Dict[str, float],
        market_id: str,
    ) -> tuple[bool, str]:
        """
        Check if bet is eligible given current portfolio

        Args:
            bet_size: Proposed bet size
            bankroll: Current bankroll
            current_positions: Dict of market_id -> position_value
            market_id: Market ID for this bet

        Returns:
            (is_eligible, reason)
        """
        # Calculate current exposure to this market
        current_market_exposure = current_positions.get(market_id, 0.0)
        new_total_exposure = current_market_exposure + bet_size

        # Check position size limit (Total exposure vs Bankroll)
        if new_total_exposure > bankroll * self.max_position_size:
            return False, f"Bet exceeds max position size. New total: {new_total_exposure:.2f} > Limit: {bankroll * self.max_position_size:.2f}"

        # Check portfolio exposure limit
        # Sum of all OTHER positions + new total for this position
        other_positions_exposure = sum(v for k, v in current_positions.items() if k != market_id)
        total_portfolio_exposure = other_positions_exposure + new_total_exposure

        if total_portfolio_exposure > bankroll * self.max_portfolio_exposure:
            return False, f"Bet exceeds max portfolio exposure: {total_portfolio_exposure:.2f} > {bankroll * self.max_portfolio_exposure:.2f}"

        return True, "Bet eligible"

    def should_exit_position(
        self,
        entry_price: float,
        current_price: float,
        outcome: str,
    ) -> tuple[bool, str]:
        """
        Check if position should be exited

        Args:
            entry_price: Entry probability
            current_price: Current market probability
            outcome: Position outcome (YES or NO)

        Returns:
            (should_exit, reason)
        """
        # Calculate P&L
        if outcome == "YES":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / (1 - entry_price)

        # Check stop loss
        if pnl_pct <= self.stop_loss:
            return True, f"Stop loss triggered: {pnl_pct:.1%}"

        return False, "Hold position"

    def calculate_portfolio_stats(
        self,
        bankroll: float,
        positions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate portfolio statistics

        Args:
            bankroll: Current bankroll
            positions: Dict of market_id -> position_value

        Returns:
            Dict of portfolio statistics
        """
        total_exposure = sum(abs(v) for v in positions.values())
        exposure_pct = total_exposure / bankroll if bankroll > 0 else 0

        return {
            "bankroll": bankroll,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "num_positions": len(positions),
            "available_capital": bankroll - total_exposure,
        }
