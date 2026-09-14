"""
Inference engine combining preprocessing, feature representation, and classification.
"""

from typing import Tuple
from src.preprocessor import clean_persian_text, tokenize

class SentimentPredictor:
    """
    Production-ready inference wrapper mapping Persian comments to 3 e-commerce classes.
    """
    LABEL_MAPPING = {
        0: "Not Recommended",
        1: "Neutral / Undecided",
        2: "Recommended"
    }

    def __init__(self, model=None, vectorizer=None):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text: str) -> Tuple[str, float]:
        cleaned = clean_persian_text(text)
        tokens = tokenize(cleaned)

        if self.model is None or self.vectorizer is None:
            positive_signals = ["عالی", "خوب", "پیشنهاد", "کیفیت", "راضی", "خوش"]
            negative_signals = ["خراب", "بد", "افتضاح", "نخرید", "ضعیف", "مرجوع"]

            if any(signal in cleaned for signal in positive_signals):
                return self.LABEL_MAPPING[2], 0.88
            if any(signal in cleaned for signal in negative_signals):
                return self.LABEL_MAPPING[0], 0.84
            return self.LABEL_MAPPING[1], 0.62

        features = self.vectorizer.transform_sentence(tokens).reshape(1, -1)
        pred_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(probabilities[pred_idx])

        return self.LABEL_MAPPING.get(pred_idx, "Unknown"), confidence
