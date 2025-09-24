import numpy as np
import abc
import src.optimizers as opt
import src.models as models 
import src.stop_criteria as stop
from src.preprocessing import Preprocessing

class Algorithm(abc.ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        self.algorithm_observers = []
        self.optimizer_strategy = optimizer_strategy
        self.model = model
        
    def add(self, observer):
       if observer not in self.algorithm_observers:
           self.algorithm_observers.append(observer)
       else:
           print('Failed to add: {}'.format(observer))

    def remove(self, observer):
       try:
           self.algorithm_observers.remove(observer)
       except ValueError:
           print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit():
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value    

    @property
    def errors(self):
        return self._errors
    
    @errors.setter
    def errors(self, value):
        self._errors = value

    @property
    def rmse(self):
        return self._rmse
    
    @rmse.setter
    def rmse(self, value):
        self._rmse = value
    
class PLA(Algorithm):


    def fit(self, X, y, stop_criteria):
        self.notify_started()
        Xplus_one = Preprocessing.build_design_matrix(X)
        self.model.w = Preprocessing.initialize_weights(Xplus_one)
        self.calculate_errors(Xplus_one,y)
        self.calculate_rmse(Xplus_one)
        self.iteration = 0
        while True:
            self.calculate_errors(Xplus_one, y)
            self.calculate_rmse(Xplus_one)
            self.notify_iteration()
            self.optimizer_strategy.update_model(X=Xplus_one, y=y, model=self.model)
            self.iteration += 1
            if stop_criteria.isFinished(self):
                break
    def calculate_rmse(self,X):
        mse = (1/len(X)) * np.square(np.linalg.norm(self.errors))
       # rmse = np.sqrt(mse)
        self.rmse = mse
    def calculate_errors(self,X,y):
        self.errors = self.model.predict(X) - y

