import numpy as np
from .estimator import Estimator

class SGDClassifier(Estimator):
    """
    Линейный классификатор (логистическая регрессия).
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Обучает линейный классификатор на данных X и целевой переменной y.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения.
        y : array-like, shape (n_samples,)
            Целевая переменная.
        """
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_b = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_b = X
        
        self.weights = np.zeros(X_b.shape[1])

        for _ in range(self.n_iterations):
            y_pred = self._sigmoid(X_b.dot(self.weights))

            gradients = (1/n_samples) * X_b.T.dot(y_pred - y)
            self.weights -= self.learning_rate * gradients

    def predict(self, X):
        """
        Предсказывает метки классов входных данных X.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.

        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные метки классов.
        """
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X
        
        y_pred = self._sigmoid(X_b.dot(self.weights))
        
        return y_pred
