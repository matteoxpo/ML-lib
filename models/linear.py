import numpy as np
import pandas as pd
from .estimator import Estimator
from abc import abstractmethod

class LinearBase(Estimator):
    def __init__(self, fit_intercept=True):
        self.weights = None
        self.bias = None
        self.fit_intercept = fit_intercept
        
    @abstractmethod
    def _update_weights(self, X, y):
        pass
        
    def fit(self, X, y):
        """
        Обучает линейный регрессор на данных X и целевой переменной y.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения.
        y : array-like, shape (n_samples,)
            Целевая переменная.
        """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        X_b = pd.DataFrame(X)
        if self.fit_intercept:
            X_b.insert(0, 'Intercept', 1) 

        self._update_weights(X_b, y)                


    
class LinearRegressor(LinearBase):
    
    def _update_weights(self, X, y):
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    
    def predict(self, X):
        """
        Предсказывает значения для входных данных X.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.

        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные значения.
        """
        X_df = pd.DataFrame(X)
        if self.fit_intercept:
            X_df.insert(0, 'Intercept', 1) 

        return X_df.dot(self.weights)
    
class LinearClassifier(LinearBase):
    """
    Линейный классификатор.
    """
    def _update_weights(self, X, y):
        self.weights = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        """
        Предсказывает значения для входных данных X.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.

        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные значения.
        """
        X_df = pd.DataFrame(X)
        if self.fit_intercept:
            X_df.insert(0, 'Intercept', 1) 

        return (X_df.dot(self.weights) >= 0).astype(int)