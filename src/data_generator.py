import numpy as np
import scipy.stats as st
from typing import Tuple

class DataGenerator():
    def __init__(self,sample_size, weights,x_min, x_max, std = 1):
        self.sample_size = sample_size
        self.weights = weights
        self.x_min = x_min
        self.x_max = x_max
        self.std = std
        
    def get_data(self):
        xStep = (self.x_max - self.x_min)/self.sample_size
        x1 = np.linspace(self.x_min, self.x_max, self.sample_size).reshape(-1, 1)
        X_plus = np.hstack((np.ones((self.sample_size,1)), x1))
        y = X_plus @ self.weights
        noise = st.norm.rvs(size=(self.sample_size,1),scale=self.std)
        y = y + noise
        return x1, y