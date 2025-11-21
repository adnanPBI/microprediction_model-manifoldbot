"""Optimize model weights based on performance"""

from typing import Dict, List
from .performance_tracker import PerformanceTracker


class ModelOptimizer:
    """Optimizes model weights based on historical performance"""

    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        min_predictions: int = 20,
        adapt_weights: bool = True,
    ):
        """
        Initialize model optimizer

        Args:
            performance_tracker: Performance tracker instance
            min_predictions: Minimum predictions before adjusting weights
            adapt_weights: Whether to adapt weights automatically
        """
        self.tracker = performance_tracker
        self.min_predictions = min_predictions
        self.adapt_weights = adapt_weights

    def calculate_optimal_weights(self) -> Dict[str, float]:
        """
        Calculate optimal weights for each model based on performance

        Returns:
            Dict mapping model_name to optimal weight
        """
        if not self.adapt_weights:
            return {}

        # Get performance for all models
        all_performance = self.tracker.get_all_models_performance()

        # Filter models with sufficient data
        valid_models = {
            name: perf
            for name, perf in all_performance.items()
            if perf.get("num_predictions", 0) >= self.min_predictions
            and not perf.get("insufficient_data", False)
        }

        if not valid_models:
            return {}

        # Weight based on inverse Brier score (lower is better)
        # Use exponential weighting to emphasize differences
        weights = {}
        brier_scores = {
            name: perf["avg_brier_score"]
            for name, perf in valid_models.items()
        }

        # Convert Brier scores to weights (lower Brier = higher weight)
        # Brier score range is [0, 1], so we use (1 - brier_score)
        raw_weights = {
            name: (1 - score) ** 2  # Square to emphasize differences
            for name, score in brier_scores.items()
        }

        # Normalize weights
        total = sum(raw_weights.values())
        if total > 0:
            weights = {
                name: weight / total
                for name, weight in raw_weights.items()
            }

        return weights

    def get_weight_recommendations(self) -> Dict[str, Dict]:
        """
        Get weight recommendations with explanations

        Returns:
            Dict with model recommendations
        """
        optimal_weights = self.calculate_optimal_weights()
        all_performance = self.tracker.get_all_models_performance()

        recommendations = {}
        for model_name, weight in optimal_weights.items():
            perf = all_performance.get(model_name, {})
            recommendations[model_name] = {
                "recommended_weight": weight,
                "current_performance": perf,
                "reason": f"Based on {perf.get('num_predictions', 0)} predictions, "
                         f"Brier score: {perf.get('avg_brier_score', 0):.4f}, "
                         f"Accuracy: {perf.get('accuracy', 0):.2%}",
            }

        return recommendations

    def should_adjust_weights(self) -> bool:
        """
        Check if weights should be adjusted

        Returns:
            True if weights should be adjusted
        """
        if not self.adapt_weights:
            return False

        all_performance = self.tracker.get_all_models_performance()

        # Check if any model has sufficient predictions
        sufficient_data = any(
            perf.get("num_predictions", 0) >= self.min_predictions
            for perf in all_performance.values()
        )

        return sufficient_data
