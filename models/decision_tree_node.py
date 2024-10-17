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
