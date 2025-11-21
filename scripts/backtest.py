"""Backtest bot strategies on historical data"""

import asyncio
from typing import List
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import ManifoldClient, Market
from src.bot import MikhailMarketMind


async def backtest():
    """Run backtest on historical markets"""
    print("=" * 60)
    print("MikhailMarketMind Backtest")
    print("=" * 60)

    client = ManifoldClient()

    # Get resolved markets from target user
    target_user = os.getenv("TARGET_USER", "MikhailTal")
    print(f"\nFetching resolved markets from {target_user}...")

    all_markets = client.get_markets_by_user(target_user, limit=100)
    resolved_markets = [m for m in all_markets if m.is_resolved]

    print(f"Found {len(resolved_markets)} resolved markets")

    if not resolved_markets:
        print("No resolved markets found for backtesting.")
        return

    # Initialize bot
    bot = MikhailMarketMind()

    # Track backtest results
    total_predictions = 0
    correct_predictions = 0
    total_edge = 0
    total_brier_score = 0

    print("\n" + "=" * 60)
    print("Running backtest...")
    print("=" * 60 + "\n")

    for market in resolved_markets:
        # Get prediction
        prediction = await bot.ensemble.predict(market)

        if prediction:
            total_predictions += 1

            # Calculate metrics
            actual = 1.0 if market.resolution == "YES" else 0.0
            predicted = prediction.probability

            # Brier score
            brier = (predicted - actual) ** 2
            total_brier_score += brier

            # Edge
            edge = abs(predicted - market.probability)
            total_edge += edge

            # Accuracy
            predicted_outcome = "YES" if predicted > 0.5 else "NO"
            if predicted_outcome == market.resolution:
                correct_predictions += 1

            print(f"Market: {market.question[:60]}...")
            print(f"  Predicted: {predicted:.2%} | Actual: {market.resolution} | Brier: {brier:.4f}")
            print()

    # Print summary
    print("\n" + "=" * 60)
    print("Backtest Summary")
    print("=" * 60)

    if total_predictions > 0:
        avg_brier = total_brier_score / total_predictions
        avg_edge = total_edge / total_predictions
        accuracy = correct_predictions / total_predictions

        print(f"Total predictions: {total_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Brier score: {avg_brier:.4f} (lower is better)")
        print(f"Average edge: {avg_edge:.2%}")
        print(f"\nBrier score interpretation:")
        print(f"  < 0.1: Excellent")
        print(f"  0.1-0.2: Good")
        print(f"  0.2-0.3: Fair")
        print(f"  > 0.3: Poor")
    else:
        print("No predictions generated.")


if __name__ == "__main__":
    asyncio.run(backtest())
