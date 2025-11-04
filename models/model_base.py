# models/model_functional.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Self


class BaseModel(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
