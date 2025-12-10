"""Main bot orchestration"""

import asyncio
import os
import time
from typing import List, Optional
import yaml
import structlog
from dotenv import load_dotenv

from .api import ManifoldClient, Market
from .predictors import (
    ClaudePredictor,
    GPTPredictor,
    OllamaPredictor,
    EnsemblePredictor,
    Prediction,
)
from .analyzers import (
    SentimentAnalyzer,
    MomentumAnalyzer,
    LiquidityAnalyzer,
)
from .strategy import KellyBetting, RiskManager, PortfolioManager
from .learning import PerformanceTracker, ModelOptimizer


# Load environment variables
load_dotenv()

# Setup logging
log = structlog.get_logger()


class MikhailMarketMind:
    """Main bot class orchestrating all components"""

    def __init__(self, config_path: str = "config/bot_config.yaml"):
        """
        Initialize bot

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize API client
        self.client = ManifoldClient()

        # Get target user
        self.target_user = os.getenv("TARGET_USER") or self.config["target"]["user"]
        log.info(f"Targeting markets by user: {self.target_user}")

        # Initialize predictors
        self.predictors = self._init_predictors()
        self.ensemble = EnsemblePredictor(
            predictors=self.predictors,
            strategy=self.config["models"]["ensemble"]["strategy"],
        )

        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer(self.client)
        self.momentum_analyzer = MomentumAnalyzer(
            self.client,
            lookback_hours=self.config["analysis"]["momentum"].get("lookback_hours", 24),
        )
        self.liquidity_analyzer = LiquidityAnalyzer(
            min_liquidity=self.config["risk"]["min_liquidity"],
        )

        # Initialize strategy components
        self.kelly_betting = KellyBetting(
            kelly_fraction=self.config["betting"]["kelly_fraction"],
            min_edge=self.config["thresholds"]["min_edge"],
            max_edge=self.config["thresholds"]["max_edge"],
            market_impact_adjustment=self.config["betting"]["market_impact_adjustment"],
        )

        self.risk_manager = RiskManager(
            max_portfolio_exposure=self.config["risk"]["max_portfolio_exposure"],
            max_position_size=self.config["risk"]["max_position_size"],
            min_liquidity=self.config["risk"]["min_liquidity"],
            stop_loss=self.config["risk"]["stop_loss"],
        )

        self.portfolio_manager = PortfolioManager()

        # Initialize learning components
        self.performance_tracker = PerformanceTracker()
        self.model_optimizer = ModelOptimizer(
            self.performance_tracker,
            min_predictions=self.config["learning"].get("min_bets_for_adjustment", 20),
            adapt_weights=self.config["learning"]["adapt_model_weights"],
        )

        # Get current bankroll
        self.bankroll = self._get_bankroll()

        log.info("MikhailMarketMind initialized successfully")

    def _init_predictors(self) -> List:
        """Initialize prediction models"""
        predictors = []

        # Claude predictor
        if self.config["models"]["claude"]["enabled"]:
            claude = ClaudePredictor(
                model=self.config["models"]["claude"]["model"],
                weight=self.config["models"]["claude"]["weight"],
            )
            if claude.enabled:
                predictors.append(claude)
                log.info(f"Initialized Claude predictor: {claude.name}")

        # GPT predictor
        if self.config["models"]["gpt"]["enabled"]:
            gpt = GPTPredictor(
                model=self.config["models"]["gpt"]["model"],
                weight=self.config["models"]["gpt"]["weight"],
            )
            if gpt.enabled:
                predictors.append(gpt)
                log.info(f"Initialized GPT predictor: {gpt.name}")

        # Ollama predictor
        if self.config["models"].get("ollama", {}).get("enabled", False):
            ollama = OllamaPredictor(
                model=self.config["models"]["ollama"]["model"],
                weight=self.config["models"]["ollama"]["weight"],
                endpoint=self.config["models"]["ollama"]["endpoint"],
            )
            if ollama.enabled:
                predictors.append(ollama)
                log.info(f"Initialized Ollama predictor: {ollama.name}")

        if not predictors:
            raise ValueError("No predictors enabled! Check your configuration and API keys.")

        return predictors

    def _get_bankroll(self) -> float:
        """Get current bankroll"""
        try:
            portfolio = self.client.get_portfolio()
            return portfolio.balance
        except Exception as e:
            log.warning(f"Could not fetch portfolio: {e}. Using default bankroll.")
            return 1000.0  # Default starting bankroll

    async def analyze_market(self, market: Market) -> Optional[dict]:
        """
        Fully analyze a market

        Args:
            market: Market to analyze

        Returns:
            Analysis results or None if market not suitable
        """
        # Check basic eligibility
        is_eligible, reason = self.risk_manager.check_market_eligibility(market)
        if not is_eligible:
            log.debug(f"Market not eligible: {reason}", market_id=market.id)
            return None

        # Get prediction from ensemble
        prediction = await self.ensemble.predict(market)
        if not prediction:
            log.debug("No prediction available", market_id=market.id)
            return None

        # Check confidence threshold
        if prediction.confidence < self.config["thresholds"]["min_confidence"]:
            log.debug(
                f"Prediction confidence too low: {prediction.confidence:.2%}",
                market_id=market.id,
            )
            return None

        # Analyze market characteristics
        liquidity_analysis = self.liquidity_analyzer.analyze(market)

        # Optional: sentiment and momentum (can be slow)
        sentiment_analysis = None
        momentum_analysis = None

        if self.config["analysis"]["sentiment"]["enabled"]:
            sentiment_analysis = self.sentiment_analyzer.analyze(market)

        if self.config["analysis"]["momentum"]["enabled"]:
            momentum_analysis = self.momentum_analyzer.analyze(market)

        return {
            "market": market,
            "prediction": prediction,
            "liquidity": liquidity_analysis,
            "sentiment": sentiment_analysis,
            "momentum": momentum_analysis,
        }

    async def make_decision(self, analysis: dict) -> Optional[dict]:
        """
        Make betting decision based on analysis

        Args:
            analysis: Market analysis

        Returns:
            Bet decision or None
        """
        market = analysis["market"]
        prediction = analysis["prediction"]

        # Calculate bet size
        bet_decision = self.kelly_betting.calculate_bet_size(
            prediction=prediction,
            market=market,
            bankroll=self.bankroll,
            max_bet=self.config["betting"]["max_bet"],
            min_bet=self.config["betting"]["min_bet"],
        )

        if not bet_decision:
            return None

        # Check risk management
        current_positions = self.portfolio_manager.get_position_values()
        is_eligible, reason = self.risk_manager.check_bet_eligibility(
            bet_size=bet_decision["bet_size"],
            bankroll=self.bankroll,
            current_positions=current_positions,
            market_id=market.id,
        )

        if not is_eligible:
            log.info(f"Bet not eligible: {reason}", market_id=market.id)
            return None

        return bet_decision

    async def execute_bet(self, market: Market, decision: dict, prediction: Prediction):
        """
        Execute a bet

        Args:
            market: Market to bet on
            decision: Bet decision
            prediction: Prediction used
        """
        try:
            log.info(
                f"Placing bet on: {market.question}",
                market_id=market.id,
                outcome=decision["outcome"],
                amount=decision["bet_size"],
                edge=f"{decision['edge']:.2%}",
            )

            # Place bet
            bet = self.client.place_bet(
                market_id=market.id,
                amount=decision["bet_size"],
                outcome=decision["outcome"],
            )

            # Record position
            self.portfolio_manager.add_position(
                market_id=market.id,
                outcome=decision["outcome"],
                bet_size=decision["bet_size"],
                entry_price=bet.prob_before,
                shares=bet.shares,
                prediction=prediction.probability,
                reasoning=prediction.reasoning,
            )

            # Record prediction for learning
            self.performance_tracker.record_prediction(
                market_id=market.id,
                market_question=market.question,
                model_name=prediction.model_name,
                prediction=prediction.probability,
                confidence=prediction.confidence,
                market_prob=market.probability,
                reasoning=prediction.reasoning,
            )

            # Update bankroll
            self.bankroll -= decision["bet_size"]

            log.info("Bet placed successfully", bet_id=bet.id)

        except Exception as e:
            log.error(f"Error placing bet: {e}", market_id=market.id)

    async def process_market(self, market: Market):
        """Process a single market"""
        # Skip if we already have a position
        if self.portfolio_manager.get_position(market.id):
            return

        # Analyze market
        analysis = await self.analyze_market(market)
        if not analysis:
            return

        # Make decision
        decision = await self.make_decision(analysis)
        if not decision:
            return

        # Execute bet
        await self.execute_bet(market, decision, analysis["prediction"])

    async def check_resolved_positions(self):
        """Check status of open positions and handle resolutions"""
        try:
            open_positions = self.portfolio_manager.get_all_positions()
            if not open_positions:
                return

            log.info(f"Checking {len(open_positions)} open positions for resolution")

            for market_id, position in list(open_positions.items()):
                # Fetch fresh market data
                try:
                    market = self.client.get_market(market_id)
                except Exception as e:
                    log.error(f"Failed to fetch market {market_id}: {e}")
                    continue

                # Handle resolved markets
                if market.is_resolved:
                    log.info(f"Market resolved: {market.question} -> {market.resolution}", market_id=market.id)

                    # Update performance tracker
                    resolved_yes = market.resolution == "YES"
                    self.performance_tracker.update_resolution(market.id, resolved_yes)

                    # Calculate PnL (simplified, assuming we hold to resolution)
                    # If we bet YES and it resolves YES, we win (1/entry_price - 1) * size approx
                    # Actually, if we bet YES at P, and it resolves YES (1), we get 1/P * size back. Profit = size/P - size
                    # If NO, we lose everything (assuming binary options resolving to 0 or 1)

                    # Calculate PnL for main position
                    main_pnl = self._calculate_pnl(
                        bet_size=position["bet_size"],
                        entry_price=position["entry_price"],
                        outcome=position["outcome"],
                        resolution=market.resolution
                    )

                    total_pnl = main_pnl
                    total_payout = position["bet_size"] + main_pnl

                    # Calculate PnL for hedges
                    if "hedges" in position:
                        for hedge in position["hedges"]:
                             hedge_pnl = self._calculate_pnl(
                                bet_size=hedge["bet_size"],
                                entry_price=hedge["entry_price"],
                                outcome=hedge["outcome"],
                                resolution=market.resolution
                             )
                             total_pnl += hedge_pnl
                             total_payout += hedge["bet_size"] + hedge_pnl

                    # Close position
                    self.portfolio_manager.close_position(
                        market_id=market.id,
                        exit_price=1.0 if market.resolution == "YES" else 0.0,
                        realized_pnl=total_pnl,
                        reason="market_resolved"
                    )

                    # Update bankroll
                    if total_payout > 0:
                        self.bankroll += total_payout

        except Exception as e:
            log.error(f"Error checking resolved positions: {e}")

    def _calculate_pnl(self, bet_size: float, entry_price: float, outcome: str, resolution: str) -> float:
        """Calculate PnL for a single bet"""
        if resolution == "YES":
            if outcome == "YES":
                return (bet_size / entry_price) - bet_size
            else:
                return -bet_size
        elif resolution == "NO":
            if outcome == "NO":
                return (bet_size / (1 - entry_price)) - bet_size
            else:
                return -bet_size
        else:
            return 0.0

    async def manage_positions(self):
        """Manage open positions (stop loss, take profit)"""
        try:
            open_positions = self.portfolio_manager.get_all_positions()
            if not open_positions:
                return

            for market_id, position in list(open_positions.items()):
                # Fetch fresh market data
                try:
                    market = self.client.get_market(market_id)
                except Exception as e:
                    log.error(f"Failed to fetch market {market_id}: {e}")
                    continue

                if market.is_resolved:
                    continue # Handled by check_resolved_positions

                # Check exit conditions
                should_exit, reason = self.risk_manager.should_exit_position(
                    entry_price=position["entry_price"],
                    current_price=market.probability,
                    outcome=position["outcome"]
                )

                if should_exit:
                    log.info(f"Exiting position for {market.question}: {reason}", market_id=market.id)

                    # Execute sell/exit
                    # To exit a YES position, we sell YES (or buy NO)
                    # Manifold API might have a specific 'sell' endpoint or we bet opposite.
                    # Standard API usually supports selling shares.
                    # But ManifoldClient wrapper here only has 'place_bet'.
                    # Betting NO is equivalent to selling YES if we assume sufficient liquidity/AMM.

                    # To close: Bet opposite amount approx equal to position value or uses shares mechanism.
                    # Since we don't have 'sell' in client, we'll try to hedge/close by betting opposite.
                    # This is imperfect but works for MVP.

                    exit_outcome = "NO" if position["outcome"] == "YES" else "YES"

                    # We need to calculate how much to bet to neutralize exposure?
                    # Or just close it?
                    # For simple "exit", we assume we sell the position.
                    # If client doesn't support sell, we can't truly "exit" in Manifold sense (convert to M$).
                    # We can only hedge.
                    # But the task implies "Sell order (bet on opposite outcome) to close."

                    # Let's bet the same amount on the opposite side? No, that's not closing.
                    # We want to sell our shares.
                    # If we can't sell shares via this client, we might just log it and mark as closed in portfolio manager
                    # to stop tracking it, but that leaves money on the table.
                    # The Issue description says: "Place a sell order (bet on opposite outcome) to close."

                    # So we will place a bet on the opposite outcome.
                    # How much? Rough approximation: same amount as original bet (hedge).
                    # Ideally we sell shares.

                    # Assuming we just place a bet to hedge.
                    hedge_amount = position["bet_size"]

                    try:
                         self.client.place_bet(
                            market_id=market.id,
                            amount=hedge_amount,
                            outcome=exit_outcome
                        )

                         # Calculate rough PnL (realized)
                         # This is complex without true sell.
                         # We'll mark it closed with estimated PnL based on current prob.

                         # Est PnL = (current_val - entry_cost)
                         # current_val = shares * current_price
                         # shares = bet_size / entry_price
                         shares = position["shares"]
                         current_val = shares * market.probability if position["outcome"] == "YES" else shares * (1 - market.probability)
                         realized_pnl = current_val - position["bet_size"]

                         # Record hedge in portfolio manager (keep position open to track resolution)
                         self.portfolio_manager.add_hedge_position(
                            market_id=market.id,
                            outcome=exit_outcome,
                            bet_size=hedge_amount,
                            entry_price=market.probability,
                            shares=shares, # Approx same shares
                            reasoning=reason
                         )

                         # Update bankroll (cost of hedge)
                         self.bankroll -= hedge_amount

                         log.info(f"Hedged position {market_id} with {hedge_amount} on {exit_outcome}")

                    except Exception as e:
                        log.error(f"Failed to exit position {market_id}: {e}")

        except Exception as e:
            log.error(f"Error managing positions: {e}")

    async def run_iteration(self):
        """Run one iteration of the bot"""
        try:
            log.info("Starting bot iteration")

            # 1. Manage existing positions (check resolution, exit strategies)
            await self.check_resolved_positions()
            await self.manage_positions()

            # Get markets from target user
            markets = self.client.get_markets_by_user(
                self.target_user,
                limit=self.config["execution"]["max_concurrent_markets"],
            )

            log.info(f"Found {len(markets)} markets from {self.target_user}")

            # Filter to open markets
            open_markets = [m for m in markets if not m.is_resolved]

            # Process markets concurrently
            await asyncio.gather(*[self.process_market(m) for m in open_markets])

            # Check for weight updates
            if self.model_optimizer.should_adjust_weights():
                recommendations = self.model_optimizer.get_weight_recommendations()
                log.info("Model weight recommendations:", recommendations=recommendations)

            # Log portfolio stats
            positions = self.portfolio_manager.get_position_values()
            stats = self.risk_manager.calculate_portfolio_stats(
                self.bankroll, positions
            )
            log.info("Portfolio stats:", **stats)

        except Exception as e:
            log.error(f"Error in bot iteration: {e}")

    async def run(self):
        """Main bot loop"""
        log.info("Starting MikhailMarketMind bot")

        update_interval = self.config["execution"]["update_interval"]

        while True:
            await self.run_iteration()
            log.info(f"Sleeping for {update_interval} seconds")
            await asyncio.sleep(update_interval)


async def main():
    """Main entry point"""
    bot = MikhailMarketMind()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
