import numpy as np
import pandas as pd
from typing import Self
from autogluon.tabular import TabularPredictor


class MitraImpl:
    """Minimal adapter for Mitra."""

    model_name: str = "Mitra"

    def __init__(self) -> None:
        self._model = None # do not instantiate yet

        self.isFit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        # Encode labels to integers
        classes = np.unique(y)
        self._lbl2idx = {lbl: i for i, lbl in enumerate(classes)}
        self._idx2lbl = {i: lbl for lbl, i in self._lbl2idx.items()}
        y_idx = np.array([self._lbl2idx[v] for v in y], dtype=int)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y_idx

        self._model = TabularPredictor(label="target")
        self._model.fit(df, hyperparameters={'MITRA': {'fine_tune': False}})
        self.isFit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        
        preds_idx = self._model.predict(df).to_numpy()
        return np.array([self._idx2lbl[i] for i in preds_idx], dtype=object)


class MitraRegImpl:
    """Minimal adapter for Mitra."""

    model_name: str = "Mitra"

    def __init__(self) -> None:
        self._model = None # do not instantiate yet

        self.isFit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X)
        y = np.asarray(y)

        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        self._model = TabularPredictor(label="target")
        self._model.fit(df, hyperparameters={'MITRA': {'fine_tune': False}})
        self.isFit = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.isFit:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        
        return self._model.predict(df).to_numpy()
