"""Tests for Ollama predictor"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from src.predictors.ollama_predictor import OllamaPredictor
from src.api.models import Market

@pytest.mark.asyncio
async def test_ollama_prediction_success():
    """Test successful Ollama prediction"""
    # Mock the ClientSession context manager and post method
    with patch('aiohttp.ClientSession') as MockSession:
        # Create the session mock
        session_mock = MagicMock()

        # Make ClientSession() return an async context manager that yields session_mock
        session_context = MagicMock()
        async def session_aenter(*args, **kwargs):
            return session_mock
        async def session_aexit(*args, **kwargs):
            pass
        session_context.__aenter__ = session_aenter
        session_context.__aexit__ = session_aexit
        MockSession.return_value = session_context

        # Create the response mock
        response_mock = Mock()
        response_mock.status = 200
        async def json_data():
            return {
                "response": "PROBABILITY: 0.75\nCONFIDENCE: 0.80\nREASONING: Test reasoning"
            }
        response_mock.json = json_data

        # session.post() returns an async context manager that yields response_mock
        post_context = MagicMock()
        async def post_aenter(*args, **kwargs):
            return response_mock
        async def post_aexit(*args, **kwargs):
            pass
        post_context.__aenter__ = post_aenter
        post_context.__aexit__ = post_aexit
        session_mock.post.return_value = post_context

        # Initialize predictor
        predictor = OllamaPredictor(model="llama2")
        market = Mock(spec=Market)
        market.question = "Test Question"
        market.probability = 0.5
        market.description = "Desc"
        market.time_until_close = 24.0
        market.liquidity = 100.0

        # Run
        prediction = await predictor.predict(market)

        # Verify
        assert prediction is not None
        assert prediction.probability == 0.75
        assert prediction.confidence == 0.80
        assert prediction.model_name == "ollama-llama2"

@pytest.mark.asyncio
async def test_ollama_prediction_failure():
    """Test Ollama failure handling"""
    with patch('aiohttp.ClientSession') as MockSession:
        session_mock = MagicMock()

        session_context = MagicMock()
        async def session_aenter(*args, **kwargs):
            return session_mock
        async def session_aexit(*args, **kwargs):
            pass
        session_context.__aenter__ = session_aenter
        session_context.__aexit__ = session_aexit
        MockSession.return_value = session_context

        # Simulate connection error on post
        session_mock.post.side_effect = Exception("Connection refused")

        predictor = OllamaPredictor()
        market = Mock(spec=Market)

        # Run
        prediction = await predictor.predict(market)

        # Verify it returns None instead of crashing
        assert prediction is None
