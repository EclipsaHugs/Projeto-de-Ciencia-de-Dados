import src.algorithms as al
import abc 
import matplotlib.pyplot as plt

class AlgorithmAnalyzer(abc.ABC):

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def notify_started(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_finished(self, alg: al.Algorithm):
        pass

    @abc.abstractmethod
    def notify_iteration(self, alg: al.Algorithm):
        pass
    
    
class PlotterAlgorithmObserver(AlgorithmAnalyzer):
    def init(self):
        super().init()

    def notify_finished(self, alg: al.Algorithm):
        self.iterations.append(alg.iteration)
        self.errors.append(alg.rmse)
        self.weights.append(alg.model.w)

    def notify_started(self, alg: al.Algorithm):
        self.errors = []
        self.weights = []
        self.iterations = []

    def notify_iteration(self, alg: al.Algorithm):
        self.iterations.append(alg.iteration)
        self.errors.append(alg.rmse)
        self.weights.append(alg.model.w)
 
    def plot(self,weights):
        fig, ax1 = plt.subplots()
        color='tab:red'
        plt.axhline(y=1, color=color, linestyle = 'dashed')
        ax1.set(xlabel='Iterations (t)', ylabel='RMSE(t)')
        ax1.plot(self.iterations, self.errors,color=color,label='error')
        ax1.tick_params(axis='y')
        ax1.legend(loc='center left')

        color = 'tab:green'
        ax2 = ax1.twinx()
        ax2.axhline(y=weights[0][0], color=color, linestyle = 'dashed')
        ax2.set(ylabel='Weights w(t)')
        ax2.plot(self.iterations,[w[0] for w in self.weights],color=color,label='w0')
        color = 'tab:blue'
        ax2.axhline(y=weights[1][0], color=color, linestyle = 'dashed')
        ax2.plot(self.iterations,[w[1] for w in self.weights],color=color,label='w1')
        ax2.legend(loc='center right')
        plt.show()