import numpy as np
import pandas as pd
from .estimator import Estimator

class LinearRegressor(Estimator):
    """
    Линейный классификатор.
    """

    def __init__(self, fit_intercept=True):
        self.weights = None
        self.bias = None
        self.fit_intercept = fit_intercept

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
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = np.copy(X)
            
        self.weights = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

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