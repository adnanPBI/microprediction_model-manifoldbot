"""Tests for API client"""

import pytest
from src.api.models import Market, Bet, User


def test_market_model():
    """Test Market model"""
    market_data = {
        "id": "test123",
        "creator_username": "testuser",
        "creator_name": "Test User",
        "created_time": 1234567890,
        "question": "Will this test pass?",
        "url": "https://manifold.markets/test",
        "probability": 0.75,
        "volume": 100.0,
        "is_resolved": False,
        "outcome_type": "BINARY",
    }

    market = Market(**market_data)
    assert market.id == "test123"
    assert market.is_binary
    assert market.probability == 0.75


def test_bet_model():
    """Test Bet model"""
    bet_data = {
        "id": "bet123",
        "user_id": "user123",
        "contract_id": "market123",
        "created_time": 1234567890,
        "amount": 50.0,
        "outcome": "YES",
        "shares": 60.0,
        "prob_before": 0.7,
        "prob_after": 0.72,
    }

    bet = Bet(**bet_data)
    assert bet.amount == 50.0
    assert bet.outcome == "YES"


def test_user_model():
    """Test User model"""
    user_data = {
        "id": "user123",
        "username": "testuser",
        "name": "Test User",
        "created_time": 1234567890,
    }

    user = User(**user_data)
    assert user.username == "testuser"
