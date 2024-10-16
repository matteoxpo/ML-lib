from .estimator import Estimator
import numpy as np
from utils import mean_squared_error

class SGDRegressor(Estimator):
    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
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
            
        for _ in range(self.n_iterations):
            y_pred = X_b.dot(self.weights)
            
            gradients = -(2/n_samples) * X_b.T.dot(y - y_pred)
            self.weights -= self.learning_rate * gradients

    def predict(self, X):
        """
        Предсказывает метки классов или значения для входных данных X.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.

        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные метки классов или значения.
        """
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X
        return X_b.dot(self.weights)