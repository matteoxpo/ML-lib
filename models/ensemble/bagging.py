import numpy as np
from models import DecisionTreeRegressor, DecisionTreeClassifier, Estimator
from abc import abstractmethod
from utils import sigmoid

class BaggingBase(Estimator):
    def __init__(self, n_estimators=100, estimator_class=None, **estimator_params):
        """
        n_estimators: количество моделей для бустинга
        estimator_class: класс модели, который будет использован (например, DecisionTreeClassifier)
        estimator_params: параметры, которые будут переданы в каждую модель (например, max_depth для дерева)
        """
        self.n_estimators = n_estimators
        self.estimator_class = estimator_class
        self.estimator_params = estimator_params
        self.estimators = []
        
    def fit(self, X, y):
        n_samples = X.shape[0] 
        for _ in range(self.n_estimators):
            
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]  
            y_bootstrap = y[bootstrap_indices] 
            
            estimator = self.estimator_class(**self.estimator_params)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(estimator)
            
    def predict(self, X):
        """Предсказывает значения для набора данных."""
        if X.ndim == 1:
            return self._predict_sample(X)
        return np.array([self._predict_sample(sample) for sample in X])
    
    @abstractmethod        
    def _predict_sample(self, X):
        pass

class BaggingBaseClassificator(BaggingBase):
    def __init__(self, n_estimators=100, g_threshold = 0.5, g_probabilities = False, estimator_class=None, **estimator_params):
        super().__init__(n_estimators=n_estimators, estimator_class=estimator_class, **estimator_params)
        self.g_probabilities = g_probabilities
        self.g_threshold = g_threshold
    
    def _predict_sample(self, X):
        predictions = []
        
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        y_pred = np.mean(predictions)
        
        if self.g_probabilities:
            return y_pred
        return y_pred > self.g_threshold
    
class BaggingBaseRegressor(BaggingBase):
    def _predict_sample(self, X):
        predictions = []
        
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
        y_pred = np.mean(predictions)
        
        return y_pred


class BaggingDecisionTreeClassifier(BaggingBaseClassificator):
    
    def __init__(self, n_estimators=100, max_depth = 10, g_threshold = 0.5, g_probabilities = False, min_samples_split = 2 ):
        super().__init__(n_estimators=n_estimators,
                         g_probabilities = g_probabilities, g_threshold = g_threshold, 
                         estimator_class=DecisionTreeClassifier, 
                            max_depth = max_depth, probabilities = True, min_samples_split = min_samples_split)

class BaggingDecisionTreeRegressor(BaggingBaseRegressor):
    def __init__(self, n_estimators=100, max_depth = 3, min_samples_split= 2):
        super().__init__(n_estimators=n_estimators, 
                         estimator_class=DecisionTreeRegressor, 
                            max_depth = max_depth, min_samples_split = min_samples_split)

class RandomForestClassifier(BaggingBaseClassificator):
    def __init__(self, n_estimators=100, max_depth=10, g_threshold=0.5, g_probabilities=False, max_features=None, min_samples_split=2):
        """
        n_estimators: количество деревьев
        max_depth: максимальная глубина деревьев
        g_threshold: порог для классификации
        g_probabilities: возвращать вероятности или предсказания
        max_features: количество случайных признаков для каждого разбиения узла
        min_samples_split: минимальное количество образцов для разбиения узла
        """
        super().__init__(n_estimators=n_estimators, g_threshold=g_threshold, g_probabilities=g_probabilities,
                         estimator_class=DecisionTreeClassifier, max_depth=max_depth, 
                         min_samples_split=min_samples_split, probabilities=True)
        self.max_features = max_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.estimators = []

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            max_features = self.max_features or n_features
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_bootstrap_subset = X_bootstrap[:, feature_indices]

            estimator = self.estimator_class(max_depth=self.estimator_params['max_depth'],
                                             min_samples_split=self.estimator_params['min_samples_split'])
            estimator.fit(X_bootstrap_subset, y_bootstrap)
            self.estimators.append((estimator, feature_indices))

    def _predict_sample(self, X):
        predictions = []

        for estimator, feature_indices in self.estimators:
            X_subset = X[feature_indices]
            predictions.append(estimator.predict(X_subset))

        return np.mean(predictions) if self.g_probabilities else np.round(np.mean(predictions))


class RandomForestRegressor(BaggingBaseRegressor):
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=2, max_features=None):
        """
        n_estimators: количество деревьев
        max_depth: максимальная глубина деревьев
        min_samples_split: минимальное количество образцов для разбиения узла
        max_features: количество случайных признаков для каждого разбиения узла
        """
        super().__init__(n_estimators=n_estimators,
                         estimator_class=DecisionTreeRegressor,
                         max_depth=max_depth, min_samples_split=min_samples_split)
        self.max_features = max_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.estimators = []
        
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap, y_bootstrap = X[bootstrap_indices], y[bootstrap_indices]

            max_features = self.max_features or n_features
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            X_bootstrap_subset = X_bootstrap[:, feature_indices]

            estimator = self.estimator_class(max_depth=self.estimator_params['max_depth'],
                                             min_samples_split=self.estimator_params['min_samples_split'])
            estimator.fit(X_bootstrap_subset, y_bootstrap)
            self.estimators.append((estimator, feature_indices))

    def _predict_sample(self, X):
        predictions = []
        
        for estimator, feature_indices in self.estimators:
            X_subset = X[feature_indices]
            predictions.append(estimator.predict(X_subset))
        
        return np.mean(predictions)
