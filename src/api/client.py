"""Manifold Markets API Client"""

import os
import time
from typing import List, Optional, Dict, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import Market, Bet, Comment, User, Portfolio


class ManifoldClient:
    """Client for Manifold Markets API"""

    BASE_URL = "https://api.manifold.markets/v0"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Manifold Markets API client

        Args:
            api_key: Manifold Markets API key (or from MANIFOLD_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("MANIFOLD_API_KEY")

        # Setup session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _headers(self, authenticated: bool = False) -> Dict[str, str]:
        """Get request headers"""
        headers = {"Content-Type": "application/json"}
        if authenticated and self.api_key:
            headers["Authorization"] = f"Key {self.api_key}"
        return headers

    def _get(self, endpoint: str, authenticated: bool = False, **params) -> Any:
        """Make GET request"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(
            url,
            headers=self._headers(authenticated),
            params=params,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Any:
        """Make authenticated POST request"""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.post(
            url,
            headers=self._headers(authenticated=True),
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    # User endpoints

    def get_user(self, username: str) -> User:
        """Get user by username"""
        data = self._get(f"user/{username}")
        return User(**data)

    def get_user_by_id(self, user_id: str) -> User:
        """Get user by ID"""
        data = self._get(f"user/by-id/{user_id}")
        return User(**data)

    # Market endpoints

    def get_markets(
        self,
        limit: int = 100,
        before: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Market]:
        """
        Get markets

        Args:
            limit: Number of markets to return (max 1000)
            before: Get markets before this ID
            user_id: Filter by creator user ID
        """
        params = {"limit": limit}
        if before:
            params["before"] = before
        if user_id:
            params["userId"] = user_id

        data = self._get("markets", **params)
        return [Market(**m) for m in data]

    def get_market(self, market_id: str) -> Market:
        """Get market by ID"""
        data = self._get(f"market/{market_id}")
        return Market(**data)

    def get_markets_by_user(self, username: str, limit: int = 100) -> List[Market]:
        """Get all markets created by a user"""
        user = self.get_user(username)
        return self.get_markets(limit=limit, user_id=user.id)

    # Bet endpoints

    def get_bets(
        self,
        market_id: Optional[str] = None,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        limit: int = 100,
    ) -> List[Bet]:
        """
        Get bets

        Args:
            market_id: Filter by market ID
            user_id: Filter by user ID
            username: Filter by username
            limit: Number of bets to return
        """
        params = {"limit": limit}
        if market_id:
            params["contractId"] = market_id
        if user_id:
            params["userId"] = user_id
        if username:
            params["username"] = username

        data = self._get("bets", **params)
        return [Bet(**b) for b in data]

    def place_bet(
        self,
        market_id: str,
        amount: float,
        outcome: str = "YES",
        limit_prob: Optional[float] = None,
    ) -> Bet:
        """
        Place a bet on a market

        Args:
            market_id: Market ID
            amount: Amount to bet in mana
            outcome: "YES" or "NO"
            limit_prob: Optional limit probability for the bet
        """
        data = {
            "contractId": market_id,
            "amount": amount,
            "outcome": outcome,
        }
        if limit_prob is not None:
            data["limitProb"] = limit_prob

        result = self._post("bet", data)
        return Bet(**result)

    def sell_shares(
        self,
        market_id: str,
        outcome: str,
        shares: Optional[float] = None,
    ) -> Any:
        """
        Sell shares in a market

        Args:
            market_id: Market ID
            outcome: "YES" or "NO" (the outcome you hold shares in)
            shares: Number of shares to sell (None = sell all)
        """
        data = {
            "contractId": market_id,
            "outcome": outcome,
        }
        if shares is not None:
            data["shares"] = shares

        # Note: Manifold API endpoint for selling is /v0/market/{marketId}/sell
        return self._post(f"market/{market_id}/sell", data)

    # Comment endpoints

    def get_comments(self, market_id: str) -> List[Comment]:
        """Get comments for a market"""
        data = self._get(f"comments", contractId=market_id)
        return [Comment(**c) for c in data]

    def post_comment(self, market_id: str, text: str) -> Comment:
        """Post a comment on a market"""
        data = {"contractId": market_id, "content": text}
        result = self._post("comment", data)
        return Comment(**result)

    # Portfolio endpoints

    def get_portfolio(self, username: Optional[str] = None) -> Portfolio:
        """
        Get portfolio information

        Args:
            username: Username (defaults to authenticated user)
        """
        endpoint = "me" if not username else f"user/{username}"
        data = self._get(endpoint, authenticated=True)

        balance = data.get("balance", 0)
        total_deposits = data.get("totalDeposits", 0)

        # Calculate investment value from positions
        # Note: This is simplified, actual calculation would need current market prices
        # For now, we utilize totalNetWorth from API if available to estimate investment value
        investment_value = 0
        try:
             # Ideally we would sum up open positions values.
             # Since we don't have a direct way to distinguish open/closed bets without extensive queries,
             # we rely on the API's 'totalNetWorth' if available.
             if "totalNetWorth" in data:
                 investment_value = max(0, float(data["totalNetWorth"]) - balance)
        except Exception:
            pass

        return Portfolio(
            user_id=data["id"],
            username=data["username"],
            balance=balance,
            total_deposits=total_deposits,
            investment_value=investment_value,
            total_value=balance + investment_value,
            profit=balance + investment_value - total_deposits,
            profit_percent=(balance + investment_value - total_deposits) / max(total_deposits, 1) * 100,
        )
