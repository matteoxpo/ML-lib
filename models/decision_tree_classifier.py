import numpy as np
from .estimator import Estimator
from .decision_tree_node import Node


class DecisionTreeClassifier(Estimator):
    """
    Классификатор на основе дерева решений.
    """

    def __init__(self, criterion='gini', splitter='best', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, random_state=None):
        """
        Инициализация параметров дерева решений.
        
        Параметры:
        max_depth : int, default=None
            Максимальная глубина дерева.
        min_samples_split : int, default=2
            Минимальное количество выборок для разделения.
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        """
        Обучает дерево решений на данных X и целевой переменной y.
        
        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения.
        y : array-like, shape (n_samples,)
            Целевая переменная (метки классов).
        """
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных X.
        
        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.
        
        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные метки классов.
        """
        predictions = [self._predict_sample(sample) for sample in X]
        return np.array(predictions)

    

    def _build_tree(self, X, y, depth=0):
        """
        Рекурсивно строит дерево решений.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для обучения.
        y : array-like, shape (n_samples,)
            Целевая переменная.
        depth : int, optional (default=0)
            Текущая глубина дерева.

        Возвращает:
        Node
            Корневой узел построенного дерева.
        """
        if len(np.unique(y)) == 1:
            return Node(predicted_value=y[0])
        
        if depth >= self.max_depth:
            return Node(predicted_value=self._most_common_label(y))
        
        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_gain == 0:
            return Node(predicted_value=self._most_common_label(y))

        node = Node(feature_index=best_feature, threshold=best_threshold)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        node.left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        node.right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return node
    
    def _most_common_label(self, y):
        """
        Возвращает наиболее часто встречающуюся метку.
        
        Параметры:
        y : array-like, shape (n_samples,)
            Целевая переменная.
        
        Возвращает:
        int или float
            Наиболее часто встречающаяся метка.
        """
        return np.bincount(y).argmax()
    
    def _predict_sample(self, sample):
        node = self.tree 

        while node is not None:
            if node.left is None and node.right is None:
                return node.predicted_value

            # Делите выборку по порогу
            if sample[node.feature_index] <= node.threshold:
                node = node.left  
            else:
                node = node.right 
        raise Exception('Разработчи обшибся я не знаю что в этой ситуации делать')

    def _best_split(self, X, y):
        best_impurity = float('inf')
        best_gain = 0
        best_feature_index = None
        best_threshold = None
        
        
        n_samples, n_features = X.shape
        
        # жадный поиск наилучшего критерия для разбиения
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue 
                
           
                impurity_left = self._calculate_impurity(y[left_indices])
                impurity_right = self._calculate_impurity(y[right_indices])
                
                impurity = (np.sum(left_indices) / n_samples) * impurity_left + \
                            (np.sum(right_indices) / n_samples) * impurity_right
                
                gain = self._calculate_impurity(y) - impurity
                
                if impurity < best_impurity:
                    best_impurity = impurity
                    
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold
    
        return best_feature_index, best_threshold, best_gain

    def _calculate_impurity(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
    
        if self.criterion == 'gini':
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == 'entropy':
            return -1 * (np.sum(probabilities * np.log2(probabilities)))
        else:
            raise Exception('Критерий должен быть gini или entropy')