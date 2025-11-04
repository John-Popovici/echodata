import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Self


class TabPFNImpl:
    """Minimal adapter for TabPFNClassifier."""

    model_name: str = "TabPFN"

    def __init__(self) -> None:
        self._model = TabPFNClassifier()  # use all default parameters
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
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        preds_idx = self._model.predict(X)
        return np.array([self._idx2lbl[i] for i in preds_idx], dtype=object)


class TabPFNRegImpl:
    """Minimal adapter for TabPFNRegressor."""

    model_name: str = "TabPFN"

    def __init__(self) -> None:
        self._model = TabPFNRegressor()  # use all default parameters

        self.isFit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        self._model.fit(X, y)
        self.isFit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        return self._model.predict(X)
