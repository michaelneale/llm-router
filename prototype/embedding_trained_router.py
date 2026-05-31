"""
Option B: Pre-trained frozen embedder + small trained classifier head.

- Embedder: all-MiniLM-L6-v2 (~22.7M params), FROZEN, CPU. Pre-trained by others.
- Head: sklearn LogisticRegression on the 384-d embeddings (+ has_image feature).
  Tiny (a few thousand floats), trains in milliseconds.

This is the "distill into a small head on top of strong pretrained features" point.
It needs labels to train the head, so it is NOT zero-shot for new routes — but it is
far more accurate than raw description-matching, and still GPU-free.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class EmbeddingTrainedRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.clf = None
        self.classes_ = None

    def _embed(self, texts, images):
        E = self.embedder.encode(list(texts), normalize_embeddings=True)
        img_col = np.array(images, dtype=float).reshape(-1, 1)
        return np.hstack([E, img_col])

    def fit(self, texts, images, labels):
        X = self._embed(texts, images)
        self.clf = LogisticRegression(max_iter=1000, C=10.0)
        self.clf.fit(X, labels)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, text, has_image=False):
        X = self._embed([text], [1.0 if has_image else 0.0])
        return self.clf.predict(X)[0]

    def head_params(self):
        # classes x (features+1 bias)
        coef = self.clf.coef_
        return coef.size + self.clf.intercept_.size

    def embedder_params(self):
        return sum(p.numel() for p in self.embedder.parameters())
