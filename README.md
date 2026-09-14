# Persian E-Commerce Sentiment & Purchase Recommendation Pipeline

[![CI Pipeline](https://github.com/YOUR_USERNAME/persian-ecommerce-sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/persian-ecommerce-sentiment/actions)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![UI-Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-oriented Natural Language Processing pipeline designed to classify customer reviews from Iranian e-commerce platforms into three actionable outcomes: **Recommended**, **Not Recommended**, and **Neutral / Undecided**.

---

## Background & Business Context
Customer reviews in Persian e-commerce platforms present distinct challenges:
- Complex morphology and right-to-left (RTL) script formatting.
- Inconsistent Arabic/Persian character encodings (e.g., `ي` vs `ی`).
- Zero-Width Non-Joiner (ZWNJ) variations.
- Colloquial vocabulary and informal sentiment markers.

While large Transformer models (e.g., ParsBERT) offer high expressive power, their computational overhead and GPU requirements can be prohibitive for real-time, low-latency streaming environments. This project establishes an **optimized, ultra-fast baseline (<15ms latency on CPU)** using dense Word2Vec embeddings and regularized Logistic Regression.

---

## Architecture & Pipeline Flow

```text
Raw Persian Customer Review
           │
           ▼
[ Persian Preprocessing Pipeline ]
  ├── Character normalization (Arabic -> Persian standard)
  ├── Digit, URL, and noise stripping
  ├── Stopword & ZWNJ handling
  └── Tokenization
           │
           ▼
[ Semantic Embedding ]
  ├── 300-Dimensional Word2Vec representation
  └── Sentence-level mean pooling
           │
           ▼
[ Multi-Class Regularized Logistic Classifier ]
           │
           ▼
Decision: Recommended | Not Recommended | Neutral

