# models/model_functional.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Self


class BaseModel(ABC):

    model_name: str
    model_possible_tasks: list[str]
    model_task: str
    _model: Any
    _is_fit: bool = False

    @abstractmethod
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        if model_task not in self.model_possible_tasks:
            raise ValueError(
                f"Invalid task '{model_task}' for model '{self.model_name}'. "
                + f"Allowed: {self.model_possible_tasks}"
            )
        self.model_task = model_task
        self._is_fit = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame: ...
