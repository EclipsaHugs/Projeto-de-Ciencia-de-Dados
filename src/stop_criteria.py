from __future__ import annotations
from typing import List
import src.algorithms as al

import abc

class StopCriteria(abc.ABC):

    
    @abc.abstractmethod
    def isFinished(self, alg: al.Algorithm) -> bool:
        """Implement stop criterium"""

class MaxIterationStopCriteria(StopCriteria):
    def __init__(self, max_iteration):
        self.max_iteration = max_iteration

        def isFinished(self, alg: al.Algorithm) -> bool:
            return alg.iteration >= self.max_iteration
        
class MinErrorStopCriteria(StopCriteria):

    def __init__(self, min_error):
        self.min_error = min_error

        def isFinished(self, alg: al.Algorithm) -> bool:
            return alg.rmse <= self.min_error
        
class CompositeStopCriteria(StopCriteria):
    def __init__(self):
        self.stop_criterias = []
        super().__init__()

        def add(self, stop_criteria):
            self.stop_criterias.append(stop_criteria)

        def isFinished (self, alg: al.Algorithm) -> bool:
            return any ([s.isFinished(alg) for s in self.stop_criterias])
            
    
