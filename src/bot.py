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
        self.target_user = self.config["target"]["user"]
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

    async def run_iteration(self):
        """Run one iteration of the bot"""
        try:
            log.info("Starting bot iteration")

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
