"""
Sentence-level embedding utilities leveraging pre-trained Word2Vec representations.
"""

from typing import List
import numpy as np

class MeanEmbeddingVectorizer:
    """
    Transforms tokenized sentences into fixed-size dense representations
    by calculating the centroid (mean-pooling) across 300-dimensional word vectors.
    """
    def __init__(self, word2vec_model=None, embedding_dim: int = 300):
        self.word2vec_model = word2vec_model
        self.dim = embedding_dim

    def transform_sentence(self, tokens: List[str]) -> np.ndarray:
        if not tokens or self.word2vec_model is None:
            return np.zeros(self.dim, dtype=np.float32)

        vectors = [
            self.word2vec_model[word]
            for word in tokens
            if word in self.word2vec_model
        ]

        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)

        return np.mean(vectors, axis=0)
