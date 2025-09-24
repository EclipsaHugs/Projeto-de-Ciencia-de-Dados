from tokenize import Double
import numpy as np
import scipy.stats as st
import abc
import src.models as models

class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""
    

class SteepestDescentMethod(OptimizerStrategy):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

        def update_model(sexlf, X, y, model):
            N = len(X)
            dedx = 2.0/N * (X.T @ X @ model.w - X.T @ y)
            model.w = model.w - self.learning_rate * dedx