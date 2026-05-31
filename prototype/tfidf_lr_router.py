"""
Tiny TF-IDF + multinomial logistic regression router, pure-stdlib (no numpy/sklearn).

This is the "distill into a small CPU classifier" option from the design
discussion, implemented with zero ML dependencies so it runs anywhere.

Pipeline:
  text -> word/char features (unigrams + bigrams + has_image flag)
       -> TF-IDF weighting
       -> softmax linear classifier (trained by gradient descent)
       -> intent label

Everything is plain Python dicts/lists. Model size = vocabulary x classes
floats, which we report at the end.
"""
import math
import re
import random
from collections import defaultdict, Counter

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    toks = TOKEN_RE.findall((text or "").lower())
    grams = list(toks)
    # add bigrams for a little context
    for i in range(len(toks) - 1):
        grams.append(toks[i] + "_" + toks[i + 1])
    return grams


def featurize(text, has_image):
    feats = Counter(tokenize(text))
    if has_image:
        feats["<HAS_IMAGE>"] += 1  # metadata signal as a feature
    return feats


class TfidfLogReg:
    def __init__(self):
        self.idf = {}
        self.vocab = {}          # term -> index
        self.classes = []        # list of intent labels
        self.W = []              # classes x (vocab+1 bias) weight matrix

    # ---------- TF-IDF ----------
    def _fit_idf(self, docs_feats):
        n = len(docs_feats)
        df = defaultdict(int)
        for f in docs_feats:
            for term in f:
                df[term] += 1
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
        self.vocab = {t: i for i, t in enumerate(sorted(self.idf))}

    def _vec(self, feats):
        # tf-idf, L2 normalized; returns dict index->value
        v = {}
        for term, tf in feats.items():
            if term in self.vocab:
                v[self.vocab[term]] = (1 + math.log(tf)) * self.idf[term]
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {i: x / norm for i, x in v.items()}

    # ---------- training ----------
    def fit(self, texts, images, labels, epochs=300, lr=0.5, l2=1e-4, seed=0):
        random.seed(seed)
        docs_feats = [featurize(t, im) for t, im in zip(texts, images)]
        self._fit_idf(docs_feats)
        X = [self._vec(f) for f in docs_feats]
        self.classes = sorted(set(labels))
        cidx = {c: i for i, c in enumerate(self.classes)}
        Y = [cidx[l] for l in labels]

        C = len(self.classes)
        V = len(self.vocab)
        # weights: C x (V+1), last col is bias
        self.W = [[0.0] * (V + 1) for _ in range(C)]

        idxs = list(range(len(X)))
        for ep in range(epochs):
            random.shuffle(idxs)
            for n in idxs:
                x = X[n]
                y = Y[n]
                # scores
                scores = []
                for c in range(C):
                    w = self.W[c]
                    s = w[V]  # bias
                    for i, val in x.items():
                        s += w[i] * val
                    scores.append(s)
                m = max(scores)
                exps = [math.exp(s - m) for s in scores]
                Z = sum(exps)
                probs = [e / Z for e in exps]
                # gradient step (softmax cross-entropy)
                for c in range(C):
                    err = probs[c] - (1.0 if c == y else 0.0)
                    w = self.W[c]
                    for i, val in x.items():
                        w[i] -= lr * (err * val + l2 * w[i])
                    w[V] -= lr * err
        return self

    # ---------- inference ----------
    def predict(self, text, has_image=False):
        x = self._vec(featurize(text, has_image))
        C = len(self.classes)
        V = len(self.vocab)
        best_c, best_s = 0, -1e18
        for c in range(C):
            w = self.W[c]
            s = w[V]
            for i, val in x.items():
                s += w[i] * val
            if s > best_s:
                best_s, best_c = s, c
        return self.classes[best_c]

    def num_params(self):
        return len(self.W) * (len(self.vocab) + 1)
