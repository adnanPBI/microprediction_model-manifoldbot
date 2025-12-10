"""Data models for Manifold Markets API"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class User(BaseModel):
    """Manifold Markets user"""
    id: str
    username: str
    name: str
    created_time: int
    balance: Optional[float] = None
    total_deposits: Optional[float] = None


class Market(BaseModel):
    """Manifold Markets market/contract"""
    id: str
    creator_username: str
    creator_name: str
    created_time: int
    close_time: Optional[int] = None
    question: str
    description: Optional[str] = ""
    url: str
    pool: Optional[Dict[str, float]] = None
    probability: float
    volume: float = 0
    volume_24_hours: float = 0
    is_resolved: bool = False
    resolution: Optional[str] = None
    resolution_time: Optional[int] = None
    last_updated_time: Optional[int] = None
    mechanism: str = "cpmm-1"
    outcome_type: str = "BINARY"

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary market"""
        return self.outcome_type == "BINARY"

    @property
    def liquidity(self) -> float:
        """Calculate market liquidity"""
        if self.pool and "YES" in self.pool and "NO" in self.pool:
            return min(self.pool["YES"], self.pool["NO"]) * 2
        return self.volume_24_hours

    @property
    def time_until_close(self) -> Optional[float]:
        """Hours until market closes"""
        if self.close_time:
            # Use UTC for consistency
            now = datetime.now(timezone.utc).timestamp() * 1000
            return (self.close_time - now) / (1000 * 3600)
        return None


class Bet(BaseModel):
    """A bet on a market"""
    id: str
    user_id: str
    contract_id: str
    created_time: int
    amount: float
    outcome: str
    shares: float
    prob_before: float
    prob_after: float
    fees: Optional[Dict[str, float]] = None


class Comment(BaseModel):
    """A comment on a market"""
    id: str
    user_id: str
    user_username: str
    user_name: str
    contract_id: str
    created_time: int
    text: str


class Portfolio(BaseModel):
    """User's portfolio information"""
    user_id: str
    username: str
    balance: float
    total_deposits: float
    investment_value: float
    total_value: float
    profit: float
    profit_percent: float
