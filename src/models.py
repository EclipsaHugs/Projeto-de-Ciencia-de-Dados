# src/models.py

import abc
import numpy as np

class Model(abc.ABC):
    
    def __init__(self) -> None:
        super().__init__()
        self._w = None

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

class LinearModel(Model):

    def predict(self, X: np.ndarray) -> np.ndarray:

        if self.w is None:
            raise ValueError("Os pesos (w) do modelo não foram inicializados.")
        return X @ self.w