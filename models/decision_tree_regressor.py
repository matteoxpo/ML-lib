import numpy as np
from .estimator import Estimator
from .decision_tree_node import Node
from utils.metrics import mean_squared_error, mean_absolute_error


class DecisionTreeRegressor(Estimator):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2):
        if max_depth is not None and max_depth < 1:
            raise Exception('max_depth должна быть больше либо равна 1')
        if min_samples_split < 2:
            raise Exception('min_samples_split должна быть больше либо равна 2')

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _calculate_impurity(self, y):
        """Вычисляет критерий разбиения для текущего узла"""
        if self.criterion == 'mse':
            return mean_squared_error(y, np.full(len(y), np.mean(y)))
        elif self.criterion == 'mae':
            return mean_absolute_error(y, np.full(len(y), np.mean(y)))
        else:
            raise Exception("Критерий должен быть 'mse' или 'mae'")

    def _best_split(self, X, y):
        """Находит лучшее разбиение для текущего узла"""
        best_impurity = float('inf')
        best_feature_index, best_threshold, best_gain = None, None, None
        current_impurity = self._calculate_impurity(y)

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = X[:, feature_index] >= threshold

                if np.sum(left_indices) < self.min_samples_split or np.sum(right_indices) < self.min_samples_split:
                    continue

                left_impurity = self._calculate_impurity(y[left_indices])
                right_impurity = self._calculate_impurity(y[right_indices])
                weighted_impurity = (len(y[left_indices]) * left_impurity + len(y[right_indices]) * right_impurity) / len(y)

                gain = current_impurity - weighted_impurity

                # Если текущее разбиение улучшает модель (gain > 0), мы его запоминаем
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_impurity = weighted_impurity

        return best_feature_index, best_threshold, best_gain

    def _build_tree(self, X, y, depth=0):
        """Рекурсивно строит дерево решений"""
        # Остановка по критериям: глубина дерева или минимальное количество выборок
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return Node(predicted_value=np.mean(y))

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        # Если не удалось найти хорошее разбиение, вернуть лист с предсказанием
        if best_feature is None or best_gain <= 0:
            return Node(predicted_value=np.mean(y))

        # Рекурсивно строим дерево для левой и правой ветви
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Создаем текущий узел
        return Node(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        """Обучает дерево решений"""
        self.root = self._build_tree(X, y)

    def _predict_sample(self, sample):
        """Предсказывает значение для одного образца"""
        node = self.root
        while not node.is_leaf():
            if sample[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_value

    def predict(self, X):
        """Предсказывает значения для набора данных"""
        return np.array([self._predict_sample(sample) for sample in X])
