import numpy as np
from models import DecisionTreeRegressor, DecisionTreeClassifier, Estimator
from abc import abstractmethod
from utils import sigmoid

class GradientBoostingBase(Estimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, estimator_class=None, **estimator_params):
        """
        n_estimators: количество моделей для бустинга
        learning_rate: скорость обучения
        estimator_class: класс модели, который будет использован (например, DecisionTreeClassifier)
        estimator_params: параметры, которые будут переданы в каждую модель (например, max_depth для дерева)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimators = []
        
    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y))

        for _ in range(self.n_estimators):

            estimator = self.estimator_class(**self.estimator_params)
            estimator.fit(X, y - y_pred)

            y_pred += self.learning_rate * estimator.predict(X)

            self.estimators.append(estimator)
            
    def predict(self, X):
        """Предсказывает значения для набора данных."""
        if X.ndim == 1:
            return self._predict_sample(X)
        return np.array([self._predict_sample(sample) for sample in X])
    
    @abstractmethod        
    def _predict_sample(self, X):
        pass

class GradientBoostingBaseClassificator(GradientBoostingBase):
    def __init__(self, n_estimators=100, learning_rate=0.1, g_threshold = 0.5, g_probabilities = False, estimator_class=None, **estimator_params):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, estimator_class=estimator_class, **estimator_params)
        self.g_probabilities = g_probabilities
        self.g_threshold = g_threshold
    
    def _predict_sample(self, X):
        y_pred = 0

        for tree in self.estimators:
            y_pred += self.learning_rate * tree.predict(X)
        if self.g_probabilities:
            return y_pred
        return y_pred > self.g_threshold
    
class GradientBoostingBaseRegressor(GradientBoostingBase):
    def _predict_sample(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.estimators:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


class GradientBoostingDecisionTreeClassifier(GradientBoostingBaseClassificator):
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth = 3,g_threshold = 0.5, g_probabilities = False, min_samples_split = 2 ):
        """
        n_estimators: количество моделей для бустинга
        learning_rate: скорость обучения
        estimator_class: класс модели, который будет использован (например, DecisionTreeClassifier)
        estimator_params: параметры, которые будут переданы в каждую модель (например, max_depth для дерева)
        """
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, 
                         g_probabilities = g_probabilities, g_threshold = g_threshold, 
                         estimator_class=DecisionTreeClassifier, 
                            max_depth = max_depth, probabilities = True, min_samples_split = min_samples_split)

class GradientBoostingDecisionTreeRegressor(GradientBoostingBaseRegressor):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth = 3):
        """
        n_estimators: количество моделей для бустинга
        learning_rate: скорость обучения
        estimator_class: класс модели, который будет использован (например, DecisionTreeClassifier)
        estimator_params: параметры, которые будут переданы в каждую модель (например, max_depth для дерева)
        """
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, estimator_class=DecisionTreeRegressor, max_depth = max_depth)