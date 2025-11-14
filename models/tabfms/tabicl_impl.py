import pandas as pd
from tabicl import TabICLClassifier
from typing import Any, override
from models.model_base import BaseModel
from models.model_registry import register_model


@register_model
class TabICLImpl(BaseModel):
    """Minimal implementation for TabICL."""

    model_name: str = "TabICL"
    model_possible_tasks: list[str] = [
        "classification",
    ]

    @override
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        super().__init__(model_task, **kwargs)
        self._model = TabICLClassifier()

    @override
    def _fit_impl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._model.fit(X, y)

    @override
    def _predict_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(X)
