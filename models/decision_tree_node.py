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
    
    def print_tree(self, depth=0):
        if self.is_leaf():
            # Выводим только предсказанное значение для листа
            indent = " " * (depth * 2)
            print(f"{indent}{self.predicted_value:.2f}")
        else:
            # Выводим информацию о текущем узле
            indent = " " * (depth * 2)  # Отступы в зависимости от глубины
            condition = f"[X{self.feature_index}]"
            print(f"{indent}{condition}")
            print(f"{indent} {' ' * len(condition)} |")
            print(f"{indent} ______")
            print(f"{indent} |      |")
            print(f"{indent} [<={self.threshold:.2f}]  ", end="")
            if self.left.is_leaf():
                print(f"{self.left.predicted_value:.2f}")
            else:
                print()  # Переход на новую строку для дальнейшего отображения
            print(f"{indent} [>{self.threshold:.2f}]  ", end="")
            if self.right.is_leaf():
                print(f"{self.right.predicted_value:.2f}")
            else:
                print()  # Переход на новую строку для дальнейшего отображения
            
            # Если узлы не являются листьями, выводим их
            if not self.left.is_leaf():
                self.left.print_tree(depth + 1)
            if not self.right.is_leaf():
                self.right.print_tree(depth + 1)
    
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
