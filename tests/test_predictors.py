"""Tests for prediction models"""

import pytest
from src.predictors.base import Prediction


def test_prediction_validation():
    """Test Prediction validation"""
    # Valid prediction
    pred = Prediction(probability=0.7, confidence=0.8, model_name="test")
    assert pred.probability == 0.7
    assert pred.confidence == 0.8

    # Invalid probability
    with pytest.raises(AssertionError):
        Prediction(probability=1.5, confidence=0.8, model_name="test")

    # Invalid confidence
    with pytest.raises(AssertionError):
        Prediction(probability=0.7, confidence=-0.1, model_name="test")
