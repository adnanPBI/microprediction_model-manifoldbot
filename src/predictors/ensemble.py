"""Ensemble predictor combining multiple models"""

import asyncio
from typing import List, Optional
import numpy as np

from .base import BasePredictor, Prediction
from ..api.models import Market


class EnsemblePredictor(BasePredictor):
    """Combines predictions from multiple models"""

    def __init__(
        self,
        predictors: List[BasePredictor],
        strategy: str = "weighted_average",
        weight: float = 1.0,
    ):
        """
        Initialize ensemble predictor

        Args:
            predictors: List of predictors to ensemble
            strategy: Combination strategy (weighted_average, majority_vote, confidence_weighted)
            weight: Weight for this ensemble
        """
        super().__init__(name="ensemble", weight=weight)
        self.predictors = predictors
        self.strategy = strategy

    def add_predictor(self, predictor: BasePredictor):
        """Add a predictor to the ensemble"""
        self.predictors.append(predictor)

    def remove_predictor(self, predictor_name: str):
        """Remove a predictor from the ensemble"""
        self.predictors = [p for p in self.predictors if p.name != predictor_name]

    async def predict(self, market: Market) -> Optional[Prediction]:
        """Generate ensemble prediction"""
        if not self.enabled or not self.predictors:
            return None

        # Get predictions from all enabled predictors concurrently
        tasks = [p.predict(market) for p in self.predictors if p.enabled]
        predictions = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        valid_predictions = [
            p for p in predictions
            if isinstance(p, Prediction) and p is not None
        ]

        if not valid_predictions:
            return None

        # Combine predictions based on strategy
        if self.strategy == "weighted_average":
            return self._weighted_average(valid_predictions)
        elif self.strategy == "confidence_weighted":
            return self._confidence_weighted(valid_predictions)
        elif self.strategy == "majority_vote":
            return self._majority_vote(valid_predictions)
        else:
            return self._weighted_average(valid_predictions)

    def _weighted_average(self, predictions: List[Prediction]) -> Prediction:
        """Weighted average of predictions"""
        weights = np.array([p.confidence for p in predictions])
        probabilities = np.array([p.probability for p in predictions])

        # Normalize weights
        weights = weights / weights.sum()

        ensemble_prob = (probabilities * weights).sum()
        ensemble_confidence = weights.mean()  # Average confidence

        # Collect reasoning
        reasoning_parts = [
            f"{p.model_name}: {p.probability:.2%} (conf: {p.confidence:.2%})"
            for p in predictions
        ]
        reasoning = "Ensemble prediction:\n" + "\n".join(reasoning_parts)

        return Prediction(
            probability=ensemble_prob,
            confidence=ensemble_confidence,
            reasoning=reasoning,
            model_name=self.name,
        )

    def _confidence_weighted(self, predictions: List[Prediction]) -> Prediction:
        """Confidence-weighted combination"""
        # Weight by both predictor weight and prediction confidence
        weights = np.array([
            p.confidence
            for p in predictions
        ])
        probabilities = np.array([p.probability for p in predictions])

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones_like(weights) / len(weights)

        ensemble_prob = (probabilities * weights).sum()

        # Original issue: ensemble_confidence = (weights ** 2).sum() which is usually lower than individual confidences
        # Use a better metric, e.g., weighted average of confidences
        raw_confidences = np.array([p.confidence for p in predictions])
        ensemble_confidence = (raw_confidences * weights).sum()

        reasoning_parts = [
            f"{p.model_name}: {p.probability:.2%} (conf: {p.confidence:.2%})"
            for p in predictions
        ]
        reasoning = "Confidence-weighted ensemble:\n" + "\n".join(reasoning_parts)

        return Prediction(
            probability=ensemble_prob,
            confidence=ensemble_confidence,
            reasoning=reasoning,
            model_name=self.name,
        )

    def _majority_vote(self, predictions: List[Prediction]) -> Prediction:
        """Majority vote with threshold at 0.5"""
        votes = [1 if p.probability > 0.5 else 0 for p in predictions]
        majority = sum(votes) / len(votes)

        # Average probability of majority side
        if majority > 0.5:
            probs = [p.probability for p in predictions if p.probability > 0.5]
        else:
            probs = [p.probability for p in predictions if p.probability <= 0.5]

        ensemble_prob = np.mean(probs) if probs else 0.5
        ensemble_confidence = abs(majority - 0.5) * 2  # 0 to 1 scale

        reasoning_parts = [
            f"{p.model_name}: {'YES' if p.probability > 0.5 else 'NO'} ({p.probability:.2%})"
            for p in predictions
        ]
        reasoning = f"Majority vote ({sum(votes)}/{len(votes)}):\n" + "\n".join(reasoning_parts)

        return Prediction(
            probability=ensemble_prob,
            confidence=ensemble_confidence,
            reasoning=reasoning,
            model_name=self.name,
        )
