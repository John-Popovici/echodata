import numpy as np
from tabicl import TabICLClassifier
from typing import Self


class TabICLImpl:
    """Minimal implementation of TabularFMProtocol using TabICL with default settings."""

    model_name: str = "TabICL"

    def __init__(self):
        self._model = TabICLClassifier()  # use all default parameters
        self._lbl2idx = None
        self._idx2lbl = None

        self.isFit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        # Encode labels to integers
        classes = np.unique(y)
        self._lbl2idx = {lbl: i for i, lbl in enumerate(classes)}
        self._idx2lbl = {i: lbl for lbl, i in self._lbl2idx.items()}
        y_idx = np.array([self._lbl2idx[v] for v in y], dtype=int)

        self._model.fit(X, y_idx)
        self.isFit = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not(self.isFit):
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X)
        preds_idx = self._model.predict(X)
        return np.array([self._idx2lbl[i] for i in preds_idx], dtype=object)
