import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Any, Self, override
from enum import Enum
from models.model_base import BaseModel
from models.model_registry import register_model


@register_model
class TabPFNImpl(BaseModel):
    """Minimal adapter for TabPFNClassifier."""

    model_name: str = "TabPFN"
    model_possible_tasks: list[str] = [
        "classification",
        "regression",
    ]

    @override
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        super().__init__(model_task, **kwargs)
        if self.model_task == "classification":
            self._model: Any = TabPFNClassifier()
        elif self.model_task == "regression":
            self._model = TabPFNRegressor()

    @override
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        self._model.fit(X, y)
        self._is_fit: bool = True
        return self

    @override
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(X)
