"""Base predictor interface"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from ..api.models import Market


@dataclass
class Prediction:
    """Prediction result from a model"""
    probability: float  # Predicted probability (0-1)
    confidence: float   # Model's confidence in prediction (0-1)
    reasoning: Optional[str] = None  # Optional explanation
    model_name: str = "unknown"

    def __post_init__(self):
        """Validate prediction values"""
        assert 0 <= self.probability <= 1, "Probability must be between 0 and 1"
        assert 0 <= self.confidence <= 1, "Confidence must be between 0 and 1"


class BasePredictor(ABC):
    """Base class for all prediction models"""

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize predictor

        Args:
            name: Model name
            weight: Weight for ensemble voting
        """
        self.name = name
        self.weight = weight
        self.enabled = True

    @abstractmethod
    async def predict(self, market: Market) -> Optional[Prediction]:
        """
        Generate prediction for a market

        Args:
            market: Market to predict

        Returns:
            Prediction or None if unable to predict
        """
        pass

    def disable(self):
        """Disable this predictor"""
        self.enabled = False

    def enable(self):
        """Enable this predictor"""
        self.enabled = True

    def set_weight(self, weight: float):
        """Update predictor weight"""
        assert weight >= 0, "Weight must be non-negative"
        self.weight = weight
