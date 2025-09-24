[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FSLYL6Cf)
# Ativiadade Ciência da Computação
<img src="assets/images/Unicap_Icam_Tech-01.png" alt="drawing" width="200"/>

## Identificação
**Professor**: Diego Pinheiro, PhD

**Disciplina**: Ciência dos Dados

**Atividade**: Atividade - Regressão Linear (Perceptron Learning Algorithm)

## Instruções 
> 1. Sua implementação deve estar dentro da pasta **src** 
> 2. Não modifique nenhum código dentro da pasta **test**
> 3. A submissão **não deve ser feita após o prazo** (nem 1 minuto a mais)

## Descrição da Atividade
### Questão 01
Implementa a classe `DataGenerator` no módulo `data_generator.py` para gerar no intervalo `x` [-2,2] os dados a seguir. Consulte os testes do módulo `test_01_data_generator.py`.

<img src="assets/images/plot.png" alt="drawing" width="600"/>

### Questão 02
Considere os testes no módulo `test_02_linear_model.py`, implemente o modelo linear que recebe uma matriz com `m` linhas e `d+1` colunas e retorna um vetor $\mathbf{\hat{y}}$ a seguir.

$$
\mathbf{\hat{y}} = f(X) = X \cdot \mathbf{w}
$$

Obedeça a Abstract Base Class no módulo `models.py` a seguir

```python
class Model(abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        
    @abc.abstractmethod
    def predict() -> np.ndarray:
        """Implement the predict method"""
        
    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value
```

### Questão 03
Considere os testes no módulo `test_03_optimizer.py` e implemente no módulo `optimizers.py` as estratégias de otimização `SteepestDecent` e `NewtonMethod` obedecendo a Abstract Base Class a seguir
```python
class OptimizerStrategy(abc.ABC):
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = learning_rate
    
    @abc.abstractmethod
    def update_model(self, X, y, model):
        """Implement Update Weigth Strategy"""
```


A estratégia `SteepestDecent` atualiza o peso conforme a equação a seguir:

$$
\mathbf{w}^{[t+1]} = \mathbf{w}^{[t]} - \lambda \frac{2}{N} (X^T X \mathbf{w} - X^T\mathbf{y})
$$

A estratégia `Newton Method` atualiza o peso conforme a equação a seguir:

$$
\mathbf{w}^{[t+1]} = (1 - \lambda)\mathbf{w}^{[t]} + \lambda (X^TX)^{-1}X^T\mathbf{y}
$$

### Questão 04
Considere os testes no módulo `test_04_pla.py` e implemente o algorítmo `Perceptron Learning Algorithm (PLA)` no módulo `algorithms.py` obedecendo a Abstract Base Class a seguir:

```python
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
```

### Questão 05
Considere o notebook `nb_data_analysis.ipynb` e implemente no módulo `analyzers.py` a classe `PlotterAlgorithmObserver` para criar a visualização do aprendizado do PLA a seguir

<img src="assets/images/learning.png" alt="drawing" width="600"/>

obedecendo a Abstract Base Class a seguir:

```python 
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
```