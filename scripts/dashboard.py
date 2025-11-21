"""Performance dashboard for the bot"""

import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.portfolio import PortfolioManager
from src.learning.performance_tracker import PerformanceTracker


def create_dashboard():
    """Create and display performance dashboard"""
    console = Console()

    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]MikhailMarketMind Performance Dashboard[/bold cyan]",
        border_style="cyan"
    ))

    # Initialize components
    portfolio = PortfolioManager()
    tracker = PerformanceTracker()

    # Portfolio summary
    console.print("\n[bold yellow]Portfolio Summary[/bold yellow]")
    stats = portfolio.get_performance_stats()

    portfolio_table = Table(show_header=True, header_style="bold magenta")
    portfolio_table.add_column("Metric", style="cyan")
    portfolio_table.add_column("Value", justify="right", style="green")

    portfolio_table.add_row("Total Trades", str(stats["total_trades"]))
    portfolio_table.add_row("Wins", str(stats["wins"]))
    portfolio_table.add_row("Losses", str(stats["losses"]))
    portfolio_table.add_row("Win Rate", f"{stats['win_rate']:.1%}")
    portfolio_table.add_row("Total P&L", f"{stats['total_pnl']:.2f}")
    portfolio_table.add_row("Average P&L", f"{stats['avg_pnl']:.2f}")

    if stats["wins"] > 0:
        portfolio_table.add_row("Average Win", f"{stats['avg_win']:.2f}")
    if stats["losses"] > 0:
        portfolio_table.add_row("Average Loss", f"{stats['avg_loss']:.2f}")

    console.print(portfolio_table)

    # Open positions
    console.print("\n[bold yellow]Open Positions[/bold yellow]")
    positions = portfolio.get_all_positions()

    if positions:
        positions_table = Table(show_header=True, header_style="bold magenta")
        positions_table.add_column("Market ID", style="cyan")
        positions_table.add_column("Outcome", style="yellow")
        positions_table.add_column("Bet Size", justify="right", style="green")
        positions_table.add_column("Entry Price", justify="right", style="blue")
        positions_table.add_column("Prediction", justify="right", style="magenta")

        for market_id, pos in list(positions.items())[:10]:  # Show first 10
            positions_table.add_row(
                market_id[:12] + "...",
                pos["outcome"],
                f"{pos['bet_size']:.2f}",
                f"{pos['entry_price']:.2%}",
                f"{pos['prediction']:.2%}",
            )

        console.print(positions_table)
    else:
        console.print("[dim]No open positions[/dim]")

    # Model performance
    console.print("\n[bold yellow]Model Performance[/bold yellow]")
    all_performance = tracker.get_all_models_performance()

    if all_performance:
        model_table = Table(show_header=True, header_style="bold magenta")
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Predictions", justify="right", style="yellow")
        model_table.add_column("Accuracy", justify="right", style="green")
        model_table.add_column("Brier Score", justify="right", style="blue")

        for model_name, perf in all_performance.items():
            if not perf.get("insufficient_data"):
                model_table.add_row(
                    model_name,
                    str(perf["num_predictions"]),
                    f"{perf['accuracy']:.1%}",
                    f"{perf['avg_brier_score']:.4f}",
                )

        console.print(model_table)
    else:
        console.print("[dim]No model performance data yet[/dim]")

    console.print("\n")


if __name__ == "__main__":
    create_dashboard()
