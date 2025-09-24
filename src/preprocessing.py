import numpy as np

class Preprocessing:
    
    @staticmethod
    def build_design_matrix(X: np.ndarray) -> np.ndarray:
        bias = np.ones((X.shape[0], 1))
        Xd = np.c_[bias, X] 
        return Xd