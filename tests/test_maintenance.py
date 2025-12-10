"""Tests for bot maintenance and exit logic"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.bot import MikhailMarketMind
from src.api.models import Market

@pytest.mark.asyncio
async def test_maintain_portfolio_resolutions():
    """Test that resolved markets are closed correctly"""
    # Setup
    with patch("src.bot.ManifoldClient") as MockClient, \
         patch("src.bot.PortfolioManager") as MockPortfolio, \
         patch("src.bot.PerformanceTracker") as MockTracker, \
         patch("src.bot.RiskManager") as MockRisk:

        # Setup mocks
        client = MockClient.return_value
        portfolio = MockPortfolio.return_value
        tracker = MockTracker.return_value
        risk = MockRisk.return_value

        bot = MikhailMarketMind()
        bot.client = client
        bot.portfolio_manager = portfolio
        bot.performance_tracker = tracker
        bot.risk_manager = risk

        # Mock a resolved position
        # Fix: shares should be consistent with bet_size/entry_price (100/0.5 = 200)
        portfolio.get_all_positions.return_value = {
            "mkt1": {"outcome": "YES", "shares": 200, "bet_size": 100, "entry_price": 0.5}
        }

        # Mock market response (Resolved YES)
        resolved_market = Mock(spec=Market)
        resolved_market.id = "mkt1"
        resolved_market.is_resolved = True
        resolved_market.resolution = "YES"
        resolved_market.question = "Test Market"
        client.get_market.return_value = resolved_market

        # Run
        await bot.maintain_portfolio()

        # Verify
        portfolio.close_position.assert_called_once()
        args = portfolio.close_position.call_args[1]
        assert args["realized_pnl"] > 0  # Should be a win
        assert args["reason"] == "Resolved YES"

@pytest.mark.asyncio
async def test_maintain_portfolio_stop_loss():
    """Test that stop loss triggers a sell"""
    # Setup
    with patch("src.bot.ManifoldClient") as MockClient, \
         patch("src.bot.PortfolioManager") as MockPortfolio, \
         patch("src.bot.RiskManager") as MockRisk:

        client = MockClient.return_value
        portfolio = MockPortfolio.return_value
        risk = MockRisk.return_value

        bot = MikhailMarketMind()
        bot.client = client
        bot.portfolio_manager = portfolio
        bot.risk_manager = risk

        # Mock a losing position
        portfolio.get_all_positions.return_value = {
            "mkt1": {"outcome": "YES", "shares": 10, "bet_size": 100, "entry_price": 0.8}
        }

        # Mock market response (Tanked to 0.1)
        active_market = Mock(spec=Market)
        active_market.id = "mkt1"
        active_market.is_resolved = False
        active_market.probability = 0.1
        active_market.question = "Test Market"
        client.get_market.return_value = active_market

        # Mock Risk Manager saying "SELL!"
        risk.should_exit_position.return_value = (True, "Stop loss hit")

        # Run
        await bot.maintain_portfolio()

        # Verify sell execution
        client.sell_shares.assert_called_with(
            market_id="mkt1",
            outcome="YES",
            shares=10
        )
        portfolio.close_position.assert_called_once()
