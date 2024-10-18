from abc import ABC, abstractmethod
from .estimator import Estimator
from utils import mean_squared_error, log_loss, sigmoid
import numpy as np



class SGDBase(Estimator):
    def __init__(self, 
                 learning_rate=0.01, n_iterations=1000, 
                 fit_intercept=True, 
                 batch_size=1, shuffle=True, 
                 tol=1e-4, early_stopping=True,  n_iter_no_change=5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tol = tol 
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        
    @abstractmethod 
    def _activation_function(self,z):
        pass
    
    @abstractmethod 
    def _loss_function(self,y_true, y_pred):
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

        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = np.copy(X)
            
        self.weights = np.random.randn(X_b.shape[1]) * 0.01
            
        best_loss = float('inf') 
        no_improvement_count = 0  

        for _ in range(self.n_iterations):
            if self.shuffle:
                indices = np.random.permutation(n_samples)
                X_b = X_b[indices]
                y = y[indices]

            batch_indices = range(0, n_samples, self.batch_size if self.batch_size is not None else n_samples)

            for batch_start in batch_indices:
                X_batch = X_b[batch_start:batch_start + self.batch_size]
                y_batch = y[batch_start:batch_start + self.batch_size]

                y_pred = X_batch.dot(self.weights)
                
                gradients = -(2 / X_batch.shape[0]) * X_batch.T.dot(y_batch - y_pred)
                self.weights -= self.learning_rate * gradients
                
            if self.early_stopping:
                y_pred_full = self._activation_function(X_b.dot(self.weights))
                current_loss = self._loss_function(y, y_pred_full)

                if current_loss < best_loss - self.tol:
                    best_loss = current_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count >= self.n_iter_no_change:
                    break

class SGDRegressor(SGDBase):

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
        if self.fit_intercept:
            X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_b = X
        return X_b.dot(self.weights)
    
    def _activation_function(self, z):
        return z
    
    def _loss_function(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    

import numpy as np
from .estimator import Estimator
from utils import sigmoid, log_loss

class SGDClassifier(SGDBase):
    """
    Линейный классификатор (логистическая регрессия).
    """

    def __init__(self, 
                 learning_rate=0.01, n_iterations=1000, 
                 fit_intercept=True, 
                 batch_size=1, shuffle=True, 
                 probabilities = False, threshold = 0.5, 
                 tol=1e-4, early_stopping=True,  n_iter_no_change=5):
        self.probabilities = probabilities 
        self.threshold = threshold
        super().__init__(learning_rate,n_iterations,fit_intercept,batch_size,shuffle,tol, early_stopping, n_iter_no_change)


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
        
        y_pred = sigmoid(X_b.dot(self.weights))
        if self.probabilities:
            return y_pred
        else:
            return (y_pred >= self.threshold).astype(int)

    def _activation_function(self, z):
        return sigmoid(z)
    
    def _loss_function(self, y_true, y_pred):
        return log_loss(y_true, y_pred)
    
