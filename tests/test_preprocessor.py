"""
Automated unit tests for Persian text normalization and filtering pipeline.
"""

import pytest
from src.preprocessor import clean_persian_text, tokenize

def test_arabic_persian_character_unification():
    raw_text = "كيفيت عالي و طراحي زيبا"
    expected = "کیفیت عالی و طراحی زیبا"
    assert clean_persian_text(raw_text) == expected

def test_digit_removal():
    raw_text = "Tracking code 456 for item ۱۲۳ was dispatched."
    cleaned = clean_persian_text(raw_text)
    assert "456" not in cleaned
    assert "۱۲۳" not in cleaned

def test_punctuation_and_special_characters():
    raw_text = "Does this product have a warranty?! Yes, it does."
    cleaned = clean_persian_text(raw_text)
    assert "?" not in cleaned
    assert "!" not in cleaned

def test_tokenization_output():
    raw_text = "گوشی بسیار سبک و خوش‌دست است"
    tokens = tokenize(raw_text)
    assert len(tokens) > 0
    assert isinstance(tokens, list)

def test_empty_input_handling():
    assert clean_persian_text("") == ""
    assert tokenize("") == []
