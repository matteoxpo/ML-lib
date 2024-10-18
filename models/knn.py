import numpy as np
from collections import Counter
from .estimator import Estimator
from abc import abstractmethod

class KNNBase(Estimator):
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
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
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    @abstractmethod
    def _predict(self, x):
        pass 


class KNeighborsClassifier(KNNBase):
    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, ord=self.p, axis=1)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':
            weights = 1 / distances[k_indices]
            weighted_votes = Counter()
            for label, weight in zip(k_nearest_labels, weights):
                weighted_votes[label] += weight
            return weighted_votes.most_common(1)[0][0]


class KNeighborsRegressor(KNNBase):
    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, ord=self.p, axis=1)
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.weights == 'uniform':
            return np.mean(k_nearest_labels)
        elif self.weights == 'distance':
            weights = 1 / distances[k_indices]
            weighted_sum = np.sum(k_nearest_labels * weights)
            return weighted_sum / np.sum(weights)