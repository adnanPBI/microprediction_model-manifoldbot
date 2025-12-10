"""Portfolio management and tracking"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class PortfolioManager:
    """Manages portfolio state and position tracking"""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize portfolio manager

        Args:
            data_dir: Directory to store portfolio data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.positions_file = self.data_dir / "positions.json"
        self.history_file = self.data_dir / "trade_history.json"

        self.positions: Dict[str, Dict] = self._load_positions()
        self.trade_history: List[Dict] = self._load_history()

    def _load_positions(self) -> Dict[str, Dict]:
        """Load current positions from disk"""
        if self.positions_file.exists():
            with open(self.positions_file, "r") as f:
                return json.load(f)
        return {}

    def _save_positions(self):
        """Save current positions to disk"""
        with open(self.positions_file, "w") as f:
            json.dump(self.positions, f, indent=2)

    def _load_history(self) -> List[Dict]:
        """Load trade history from disk"""
        if self.history_file.exists():
            with open(self.history_file, "r") as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save trade history to disk"""
        with open(self.history_file, "w") as f:
            json.dump(self.trade_history, f, indent=2)

    def add_position(
        self,
        market_id: str,
        outcome: str,
        bet_size: float,
        entry_price: float,
        shares: float,
        prediction: float,
        reasoning: Optional[str] = None,
    ):
        """
        Add a new position

        Args:
            market_id: Market ID
            outcome: YES or NO
            bet_size: Amount bet
            entry_price: Entry probability
            shares: Shares purchased
            prediction: Predicted probability
            reasoning: Optional reasoning for the bet
        """
        position = {
            "market_id": market_id,
            "outcome": outcome,
            "bet_size": bet_size,
            "entry_price": entry_price,
            "shares": shares,
            "prediction": prediction,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        }

        self.positions[market_id] = position
        self._save_positions()

        # Add to history
        self.trade_history.append({
            **position,
            "action": "open",
        })
        self._save_history()

    def add_hedge_position(
        self,
        market_id: str,
        outcome: str,
        bet_size: float,
        entry_price: float,
        shares: float,
        reasoning: str,
    ):
        """
        Add a hedge/exit bet to an existing position

        Args:
            market_id: Market ID
            outcome: YES or NO
            bet_size: Amount bet
            entry_price: Entry probability
            shares: Shares purchased
            reasoning: Reason for hedging
        """
        if market_id not in self.positions:
            return

        hedge_info = {
            "outcome": outcome,
            "bet_size": bet_size,
            "entry_price": entry_price,
            "shares": shares,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "type": "hedge"
        }

        # Initialize hedges list if not present
        if "hedges" not in self.positions[market_id]:
            self.positions[market_id]["hedges"] = []

        self.positions[market_id]["hedges"].append(hedge_info)
        self._save_positions()

        # Add to history
        self.trade_history.append({
            "market_id": market_id,
            **hedge_info,
            "action": "hedge_open",
        })
        self._save_history()

    def close_position(
        self,
        market_id: str,
        exit_price: float,
        realized_pnl: float,
        reason: str,
    ):
        """
        Close a position

        Args:
            market_id: Market ID
            exit_price: Exit probability
            realized_pnl: Realized profit/loss
            reason: Reason for closing
        """
        if market_id not in self.positions:
            return

        position = self.positions.pop(market_id)
        self._save_positions()

        # Add to history
        self.trade_history.append({
            **position,
            "action": "close",
            "exit_price": exit_price,
            "realized_pnl": realized_pnl,
            "reason": reason,
            "close_timestamp": datetime.now().isoformat(),
        })
        self._save_history()

    def get_position(self, market_id: str) -> Optional[Dict]:
        """Get position for a market"""
        return self.positions.get(market_id)

    def get_all_positions(self) -> Dict[str, Dict]:
        """Get all open positions"""
        return self.positions.copy()

    def get_position_values(self) -> Dict[str, float]:
        """Get position values (for risk management)"""
        return {
            market_id: pos["bet_size"]
            for market_id, pos in self.positions.items()
        }

    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics"""
        closed_trades = [t for t in self.trade_history if t.get("action") == "close"]

        if not closed_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_pnl": 0,
            }

        wins = [t for t in closed_trades if t.get("realized_pnl", 0) > 0]
        losses = [t for t in closed_trades if t.get("realized_pnl", 0) < 0]

        total_pnl = sum(t.get("realized_pnl", 0) for t in closed_trades)

        return {
            "total_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(closed_trades) if closed_trades else 0,
            "avg_win": sum(t.get("realized_pnl", 0) for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.get("realized_pnl", 0) for t in losses) / len(losses) if losses else 0,
        }
