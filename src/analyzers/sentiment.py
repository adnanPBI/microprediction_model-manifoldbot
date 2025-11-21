"""Sentiment analysis of market comments"""

from typing import List, Dict
from ..api.models import Market, Comment
from ..api.client import ManifoldClient


class SentimentAnalyzer:
    """Analyzes sentiment from market comments"""

    # Simple keyword-based sentiment (could be enhanced with ML)
    POSITIVE_KEYWORDS = {
        "yes", "definitely", "likely", "probable", "confident", "sure",
        "agree", "correct", "right", "will", "expect", "happen",
        "bullish", "optimistic", "strong", "good", "positive"
    }

    NEGATIVE_KEYWORDS = {
        "no", "unlikely", "doubtful", "improbable", "impossible", "won't",
        "disagree", "wrong", "incorrect", "never", "bearish", "pessimistic",
        "weak", "bad", "negative", "fail"
    }

    def __init__(self, client: ManifoldClient):
        """
        Initialize sentiment analyzer

        Args:
            client: Manifold API client
        """
        self.client = client

    def analyze(self, market: Market) -> Dict[str, float]:
        """
        Analyze sentiment for a market

        Args:
            market: Market to analyze

        Returns:
            Dict with sentiment metrics
        """
        try:
            comments = self.client.get_comments(market.id)
            if not comments:
                return {"sentiment_score": 0.5, "confidence": 0.0}

            sentiments = [self._analyze_comment(c) for c in comments]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5

            return {
                "sentiment_score": avg_sentiment,
                "confidence": min(len(comments) / 10.0, 1.0),  # More comments = higher confidence
                "num_comments": len(comments),
            }
        except Exception as e:
            return {"sentiment_score": 0.5, "confidence": 0.0, "error": str(e)}

    def _analyze_comment(self, comment: Comment) -> float:
        """
        Analyze single comment sentiment

        Args:
            comment: Comment to analyze

        Returns:
            Sentiment score (0 = bearish, 0.5 = neutral, 1 = bullish)
        """
        text = comment.text.lower()

        positive_count = sum(1 for word in self.POSITIVE_KEYWORDS if word in text)
        negative_count = sum(1 for word in self.NEGATIVE_KEYWORDS if word in text)

        total = positive_count + negative_count
        if total == 0:
            return 0.5

        # Normalize to 0-1 range
        sentiment = positive_count / total
        return sentiment
