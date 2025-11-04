import numpy as np
import pandas as pd
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from typing import Self


class TabSTARImpl:
    """Minimal adapter for TabSTARClassifier."""

    model_name: str = "TabSTAR"

    def __init__(self) -> None:
        self._model = TabSTARClassifier()  # use all default parameters
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

        # Conversions before TabSTAR to meet assumptions
        X, y_idx = pd.DataFrame(X), pd.Series(y_idx)
        X.columns = X.columns.map(str)

        self._model.fit(X, y_idx)
        self.isFit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)

        # Conversions before TabSTAR to meet assumptions
        X = pd.DataFrame(X)
        X.columns = X.columns.map(str)

        preds_idx = self._model.predict(X)
        return np.array([self._idx2lbl[i] for i in preds_idx], dtype=object)


class TabSTARRegImpl:
    """Minimal adapter for TabSTARRegressor."""

    model_name: str = "TabSTAR"

    def __init__(self) -> None:
        self._model = TabSTARRegressor()  # use all default parameters

        self.isFit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        # Conversions before TabSTAR to meet assumptions
        X, y_idx = pd.DataFrame(X), pd.Series(y)
        X.columns = X.columns.map(str)

        self._model.fit(X, y_idx)
        self.isFit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)

        # Conversions before TabSTAR to meet assumptions
        X = pd.DataFrame(X)
        X.columns = X.columns.map(str)

        return self._model.predict(X)
