from __future__ import annotations

from typing import Protocol, Self
import numpy as np


class TabularFMProtocol(Protocol):
    """
    Interface for tabular foundation model *classifiers*.

    Structural (duck-typed) protocol: a class "implements" this if it provides the
    same attributes/methods, without inheriting from the protocol.
    """

    # Optional self-description
    model_name: str           # short name of underlying FM (e.g. "TabICL")

    # Core API
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self : ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

