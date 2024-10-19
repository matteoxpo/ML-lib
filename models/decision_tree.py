import numpy as np
from utils.metrics import mean_squared_error, mean_absolute_error
from .estimator import Estimator
from abc import abstractmethod

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, predicted_value=None):
        """
        Инициализирует узел дерева.

        Параметры:
        feature_index : int, optional
            Индекс признака для разбиения (только для внутренних узлов).
        threshold : float, optional
            Порог для разбиения (только для внутренних узлов).
        left : Node, optional
            Левый дочерний узел.
        right : Node, optional
            Правый дочерний узел.
        predicted_value : float, optional
            Предсказанное значение (только для листового узла).
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.predicted_value = predicted_value
    
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def print_tree(self, depth=0, prefix="Root"):
        # Если это лист, то выводим предсказанное значение
        if self.is_leaf():
            indent = " " * (depth * 4)
            print(f"{indent} {prefix} -> [Leaf: {self.predicted_value:.2f}]")
        else:
            # Иначе выводим условие текущего узла
            indent = " " * (depth * 4)
            print(f"{indent} {prefix} -> [X{self.feature_index} <= {self.threshold:.2f}]")
            
            # Рекурсивно выводим левую ветку
            self.left.print_tree(depth + 1, prefix="L")
            
            # Рекурсивно выводим правую ветку
            self.right.print_tree(depth + 1, prefix="R")
    
    def max_depth(self):
        """
        Вычисляет максимальную глубину дерева, начиная с этого узла.
        
        Возвращает:
        int: Максимальная глубина дерева.
        """
        if self.is_leaf():
            return 0
        
        left_depth = self.left.max_depth() if self.left else 0
        right_depth = self.right.max_depth() if self.right else 0
        
        return 1 + max(left_depth, right_depth)

class DecisionTreeBase(Estimator):
    """
    Базовый класс для дерева решений (классификатор и регрессор).
    """

    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        """
        Инициализация параметров дерева решений.
        
        Параметры:
        criterion : str, default='gini'
            Критерий для разбиения.
        max_depth : int, default=None
            Максимальная глубина дерева.
        min_samples_split : int, default=2
            Минимальное количество выборок для разбиения.
        """
        if max_depth is not None and max_depth < 1:
            raise Exception('max_depth должна быть больше либо равна 1')
        if min_samples_split < 2:
            raise Exception('min_samples_split должна быть больше либо равна 2')

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Обучает дерево решений на данных X и целевой переменной y."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Рекурсивно строит дерево решений."""
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Node(predicted_value=self._get_leaf_value(y))

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain <= 0:
            return Node(predicted_value=self._get_leaf_value(y))

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        
        if len(X[left_indices]) < self.min_samples_split or len(X[right_indices]) < self.min_samples_split:
            return Node(predicted_value=self._get_leaf_value(y))

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y):
        """Находит лучшее разбиение для текущего узла."""
        best_gain, best_feature_index, best_threshold = None, None, None
        current_impurity = self._calculate_impurity(y)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = X[:, feature_index] >= threshold
                
                left_len = len(X[left_indices])
                right_len = len(X[right_indices]) 

                if left_len < self.min_samples_split or right_len < self.min_samples_split:
                   continue

                gain = current_impurity - self._calc_weighted_impurity(X, y, left_indices, right_indices)
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        if best_threshold is None: 
            return None, None, None

        left_len = len(X[:, best_feature_index] < best_threshold)
        right_len = len(X[:, best_feature_index] >= best_threshold)

        if left_len < self.min_samples_split or right_len < self.min_samples_split:
            return None, None, None

        return best_feature_index, best_threshold, best_gain

    @abstractmethod
    def _calculate_impurity(self, X, y):
        """Вычисляет критерий разбиения для текущего узла."""
    def _calc_weighted_impurity(self,X, y, left_indices, right_indices):
        return (len(X[left_indices]) * self._calculate_impurity(y[left_indices]) + \
                len(X[right_indices]) * self._calculate_impurity(y[right_indices])) \
                    / len(y)
    
    def _get_leaf_value(self, y):
        """Возвращает значение листа."""
        return np.mean(y)
        
    def predict(self, X):
        """Предсказывает значения для набора данных."""
        if X.ndim == 1:
            return self._predict_sample(X)
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        """Предсказывает значение для одного образца."""
        node = self.root
        while not node.is_leaf():
            if sample[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_value
        
        
class DecisionTreeRegressor(DecisionTreeBase):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        super().__init__(criterion, max_depth, min_samples_split)

    def _calculate_impurity(self, y):
        """Вычисляет критерий разбиения для текущего узла."""
        if self.criterion == 'mse':
            return mean_squared_error(y, np.full(len(y), np.mean(y)))
        elif self.criterion == 'mae':
            return mean_absolute_error(y, np.full(len(y), np.mean(y)))
        else:
            raise Exception("Критерий должен быть 'mse' или 'mae'")



class DecisionTreeClassifier(DecisionTreeBase):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, probabilities=False):
        super().__init__(criterion, max_depth, min_samples_split)
        self.probabilities = probabilities

    def fit(self, X, y):
        """Обучает дерево решений на данных X и целевой переменной y."""
        y = np.clip(y, 0, 1)
        self.root = self._build_tree(X, y)

    def _calculate_impurity(self, y):
        """Вычисляет критерий разбиения для текущего узла."""
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            return -np.sum(probabilities * np.log2(probabilities + 1e-9))
        else:
            raise Exception('Критерий должен быть gini или entropy')