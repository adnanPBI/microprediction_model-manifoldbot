"""Track prediction performance"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import numpy as np
from datetime import datetime, timezone


class PerformanceTracker:
    """Tracks and analyzes prediction performance"""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize performance tracker

        Args:
            data_dir: Directory to store performance data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.predictions_file = self.data_dir / "predictions.json"
        self.predictions: List[Dict] = self._load_predictions()

    def _load_predictions(self) -> List[Dict]:
        """Load predictions from disk"""
        if self.predictions_file.exists():
            with open(self.predictions_file, "r") as f:
                return json.load(f)
        return []

    def _save_predictions(self):
        """Save predictions to disk"""
        with open(self.predictions_file, "w") as f:
            json.dump(self.predictions, f, indent=2)

    def record_prediction(
        self,
        market_id: str,
        market_question: str,
        model_name: str,
        prediction: float,
        confidence: float,
        market_prob: float,
        reasoning: Optional[str] = None,
    ):
        """
        Record a prediction

        Args:
            market_id: Market ID
            market_question: Market question
            model_name: Name of the model
            prediction: Predicted probability
            confidence: Model confidence
            market_prob: Market probability at time of prediction
            reasoning: Optional reasoning
        """
        record = {
            "market_id": market_id,
            "market_question": market_question,
            "model_name": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "market_prob": market_prob,
            "reasoning": reasoning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": None,  # Will be updated when market resolves
            "brier_score": None,
        }

        self.predictions.append(record)
        self._save_predictions()

    def update_resolution(self, market_id: str, resolution: bool):
        """
        Update prediction with market resolution

        Args:
            market_id: Market ID
            resolution: True if resolved YES, False if resolved NO
        """
        resolution_value = 1.0 if resolution else 0.0

        for pred in self.predictions:
            if pred["market_id"] == market_id and pred["resolution"] is None:
                pred["resolution"] = resolution_value
                # Calculate Brier score: (prediction - outcome)^2
                pred["brier_score"] = (pred["prediction"] - resolution_value) ** 2

        self._save_predictions()

    def get_model_performance(
        self,
        model_name: Optional[str] = None,
        min_predictions: int = 5,
    ) -> Dict:
        """
        Get performance statistics for a model

        Args:
            model_name: Name of model (None for all models)
            min_predictions: Minimum predictions required

        Returns:
            Performance statistics
        """
        # Filter predictions
        predictions = [
            p for p in self.predictions
            if p["resolution"] is not None
            and (model_name is None or p["model_name"] == model_name)
        ]

        if len(predictions) < min_predictions:
            return {
                "model_name": model_name,
                "num_predictions": len(predictions),
                "insufficient_data": True,
            }

        # Calculate metrics
        brier_scores = [p["brier_score"] for p in predictions]
        avg_brier = np.mean(brier_scores)

        # Calibration analysis
        calibration = self._analyze_calibration(predictions)

        # Accuracy (percentage predicted correctly)
        correct = sum(
            1 for p in predictions
            if (p["prediction"] > 0.5 and p["resolution"] == 1.0)
            or (p["prediction"] <= 0.5 and p["resolution"] == 0.0)
        )
        accuracy = correct / len(predictions)

        return {
            "model_name": model_name,
            "num_predictions": len(predictions),
            "avg_brier_score": avg_brier,
            "accuracy": accuracy,
            "calibration": calibration,
        }

    def _analyze_calibration(self, predictions: List[Dict]) -> Dict:
        """
        Analyze prediction calibration

        Args:
            predictions: List of predictions with resolutions

        Returns:
            Calibration statistics
        """
        # Bin predictions
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_preds = defaultdict(list)
        binned_outcomes = defaultdict(list)

        for pred in predictions:
            bin_idx = np.digitize(pred["prediction"], bins) - 1
            bin_idx = max(0, min(bin_idx, len(bin_centers) - 1))

            binned_preds[bin_idx].append(pred["prediction"])
            binned_outcomes[bin_idx].append(pred["resolution"])

        # Calculate calibration
        calibration_data = []
        for i in range(len(bin_centers)):
            if i in binned_outcomes and len(binned_outcomes[i]) > 0:
                avg_pred = np.mean(binned_preds[i])
                avg_outcome = np.mean(binned_outcomes[i])
                count = len(binned_outcomes[i])

                calibration_data.append({
                    "bin_center": bin_centers[i],
                    "avg_prediction": avg_pred,
                    "avg_outcome": avg_outcome,
                    "count": count,
                })

        return {
            "bins": calibration_data,
            "num_bins_used": len(calibration_data),
        }

    def get_all_models_performance(self) -> Dict[str, Dict]:
        """Get performance for all models"""
        models = set(p["model_name"] for p in self.predictions)

        return {
            model: self.get_model_performance(model)
            for model in models
        }

    def get_recent_performance(
        self,
        model_name: str,
        lookback: int = 20,
    ) -> Dict:
        """
        Get recent performance for a model

        Args:
            model_name: Model name
            lookback: Number of recent predictions to analyze

        Returns:
            Recent performance statistics
        """
        # Get recent resolved predictions
        predictions = [
            p for p in self.predictions
            if p["model_name"] == model_name and p["resolution"] is not None
        ]

        # Take most recent
        recent = predictions[-lookback:] if len(predictions) > lookback else predictions

        if not recent:
            return {"num_predictions": 0, "insufficient_data": True}

        avg_brier = np.mean([p["brier_score"] for p in recent])

        correct = sum(
            1 for p in recent
            if (p["prediction"] > 0.5 and p["resolution"] == 1.0)
            or (p["prediction"] <= 0.5 and p["resolution"] == 0.0)
        )
        accuracy = correct / len(recent)

        return {
            "num_predictions": len(recent),
            "avg_brier_score": avg_brier,
            "accuracy": accuracy,
        }
