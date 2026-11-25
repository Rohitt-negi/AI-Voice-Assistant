"""
=====================================================
Module: sentiment.py
Purpose: Analyze customer message sentiment using
         a free HuggingFace model (distilbert).
         Falls back to rule-based analysis if the
         model fails to load.
=====================================================
"""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# --- Try to load the HuggingFace pipeline ---
_sentiment_pipeline = None

def _load_model():
    """
    Lazy-load the sentiment analysis model.
    Uses distilbert-base-uncased-finetuned-sst-2-english
    which is free, small (~250MB) and accurate.
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is not None:
        return True
    try:
        from transformers import pipeline
        logger.info("Loading sentiment analysis model (first run may take a minute)...")
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512
        )
        logger.info("✅ Sentiment model loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not load HuggingFace model: {e}. Using rule-based fallback.")
        return False


# --- Rule-Based Fallback Sentiment Analysis ---

POSITIVE_WORDS = {
    "great", "excellent", "amazing", "fantastic", "wonderful", "perfect",
    "love", "happy", "interested", "good", "nice", "awesome", "brilliant",
    "absolutely", "definitely", "certainly", "yes", "sure", "keen",
    "excited", "pleased", "impressive", "helpful", "useful", "affordable",
    "reasonable", "sound", "solid", "looks good", "that works", "perfect"
}

NEGATIVE_WORDS = {
    "expensive", "costly", "overpriced", "bad", "terrible", "awful",
    "disappointed", "not interested", "no", "never", "hate", "dislike",
    "problem", "issue", "concern", "difficult", "complicated", "confusing",
    "slow", "poor", "worst", "useless", "waste", "doubt", "not sure",
    "too much", "can't afford", "out of budget", "maybe not"
}


def _rule_based_sentiment(text: str) -> Tuple[str, float]:
    """
    Simple keyword-based sentiment analysis as a fallback.
    Returns (label, confidence_score).
    """
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    positive_matches = len(words.intersection(POSITIVE_WORDS))

    # Check multi-word phrases
    for phrase in POSITIVE_WORDS:
        if " " in phrase and phrase in text_lower:
            positive_matches += 1

    negative_matches = len(words.intersection(NEGATIVE_WORDS))
    for phrase in NEGATIVE_WORDS:
        if " " in phrase and phrase in text_lower:
            negative_matches += 1

    if positive_matches > negative_matches:
        score = min(0.5 + (positive_matches * 0.1), 0.95)
        return "Positive", score
    elif negative_matches > positive_matches:
        score = min(0.5 + (negative_matches * 0.1), 0.95)
        return "Negative", score
    else:
        return "Neutral", 0.5


def analyze_sentiment(text: str) -> Dict[str, any]:
    """
    Main function: analyze the sentiment of a customer message.

    Returns a dict with:
    - label: 'Positive', 'Neutral', or 'Negative'
    - score: float between 0 and 1
    - method: 'model' or 'rule-based'

    Usage:
        result = analyze_sentiment("I'm really interested in your product!")
        # {'label': 'Positive', 'score': 0.98, 'method': 'model'}
    """
    if not text or not text.strip():
        return {"label": "Neutral", "score": 0.5, "method": "rule-based"}

    # Try the HuggingFace model first
    if _load_model() and _sentiment_pipeline is not None:
        try:
            result = _sentiment_pipeline(text[:512])[0]
            # HuggingFace returns POSITIVE/NEGATIVE — map to our labels
            raw_label = result["label"]
            score = result["score"]

            if raw_label == "POSITIVE":
                label = "Positive"
            elif raw_label == "NEGATIVE":
                label = "Negative"
            else:
                label = "Neutral"

            # If model is uncertain (score < 0.6), call it Neutral
            if score < 0.6:
                label = "Neutral"
                score = 0.5

            return {"label": label, "score": round(score, 3), "method": "model"}

        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Using rule-based fallback.")

    # Fallback: rule-based
    label, score = _rule_based_sentiment(text)
    return {"label": label, "score": round(score, 3), "method": "rule-based"}


def get_sentiment_emoji(label: str) -> str:
    """Returns an emoji for the sentiment label."""
    return {"Positive": "😊", "Negative": "😟", "Neutral": "😐"}.get(label, "😐")


def get_sentiment_color(label: str) -> str:
    """Returns a hex color for the sentiment label."""
    return {
        "Positive": "#2ECC71",
        "Negative": "#E74C3C",
        "Neutral": "#F39C12"
    }.get(label, "#F39C12")
