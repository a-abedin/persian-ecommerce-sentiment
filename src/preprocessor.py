"""
Text preprocessing utilities specifically designed for Persian e-commerce customer reviews.
Handles character normalization, ZWNJ, stopword filtering, and noise removal.
"""

import re
from typing import List

DEFAULT_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای",
    "آن", "یک", "شد", "شده", "خود", "ها", "هم", "تا", "می", "بر"
}

def clean_persian_text(text: str, remove_stopwords: bool = False) -> str:
    """
    Standardize and clean Persian text:
    1. Normalize Arabic characters (ي -> ی, ك -> ک).
    2. Remove URLs, HTML tags, and non-text artifacts.
    3. Strip Latin, Persian, and Arabic digits.
    4. Remove punctuation marks while keeping Zero-Width Non-Joiner (ZWNJ).
    5. Deduplicate multiple spaces and ZWNJ characters.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[0-9\u0660-\u0669\u06F0-\u06F9]", " ", text)
    text = re.sub(r"[^\w\s\u200c]", " ", text)
    text = re.sub(r"\u200c{2,}", "\u200c", text)
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        tokens = text.split()
        tokens = [word for word in tokens if word not in DEFAULT_STOPWORDS]
        text = " ".join(tokens)

    return text

def tokenize(text: str) -> List[str]:
    """Tokenize cleaned Persian text by standard whitespaces."""
    cleaned = clean_persian_text(text)
    return cleaned.split() if cleaned else []
