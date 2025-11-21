"""Claude-based prediction model"""

import os
import re
from typing import Optional
import anthropic

from .base import BasePredictor, Prediction
from ..api.models import Market


class ClaudePredictor(BasePredictor):
    """Uses Claude to predict market outcomes"""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        weight: float = 1.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Claude predictor

        Args:
            model: Claude model to use
            weight: Weight for ensemble
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
        """
        super().__init__(name=f"claude-{model}", weight=weight)
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        if not self.api_key:
            self.enabled = False
            return

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _build_prompt(self, market: Market) -> str:
        """Build prompt for Claude"""
        prompt = f"""You are an expert forecaster analyzing prediction markets.

Market Question: {market.question}

Description: {market.description or "No additional description provided."}

Current market probability: {market.probability:.1%}
Time until close: {market.time_until_close:.1f} hours (if available)
Market liquidity: ${market.liquidity:.2f}

Your task is to provide your best estimate of the probability that this market will resolve to YES.

Consider:
1. Base rates and historical precedents
2. Available evidence and information
3. Potential biases in the current market price
4. Time horizon and resolution criteria
5. Any edge cases or ambiguities

Provide your response in this EXACT format:
PROBABILITY: <your probability as a decimal between 0 and 1>
CONFIDENCE: <your confidence in this prediction as a decimal between 0 and 1>
REASONING: <brief explanation of your prediction>

Example:
PROBABILITY: 0.65
CONFIDENCE: 0.75
REASONING: Based on historical data and current trends, I estimate a 65% chance with moderate confidence due to some uncertainty in the timeline."""

        return prompt

    def _parse_response(self, response: str) -> Optional[Prediction]:
        """Parse Claude's response"""
        try:
            # Extract probability
            prob_match = re.search(r"PROBABILITY:\s*(\d+\.?\d*)", response, re.IGNORECASE)
            if not prob_match:
                return None
            probability = float(prob_match.group(1))

            # Extract confidence
            conf_match = re.search(r"CONFIDENCE:\s*(\d+\.?\d*)", response, re.IGNORECASE)
            if not conf_match:
                confidence = 0.7  # Default confidence
            else:
                confidence = float(conf_match.group(1))

            # Extract reasoning
            reason_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|\Z)", response, re.IGNORECASE | re.DOTALL)
            reasoning = reason_match.group(1).strip() if reason_match else None

            return Prediction(
                probability=probability,
                confidence=confidence,
                reasoning=reasoning,
                model_name=self.name,
            )
        except (ValueError, AttributeError) as e:
            return None

    async def predict(self, market: Market) -> Optional[Prediction]:
        """Generate prediction using Claude"""
        if not self.enabled:
            return None

        try:
            prompt = self._build_prompt(market)

            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            print(f"Claude prediction error: {e}")
            return None
