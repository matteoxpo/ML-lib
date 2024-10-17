import unittest
import numpy as np
from models import DecisionTreeRegressor


class TestDecisionTreeRegressor(unittest.TestCase):
    
    def setUp(self):
        # Создаем тестовые данные для использования в тестах
        self.X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        self.y_train = np.array([1.1, 1.9, 3.2, 4.5, 5.0, 6.3, 7.4, 8.1, 9.0, 10.1])
        self.X_test = np.array([[1.5], [5.5], [9.5]])
    
    def test_fit(self):
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(self.X_train, self.y_train)
        
        self.assertIsNotNone(tree.root, "Дерево не было обучено")

    def test_predict(self):
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(self.X_train, self.y_train)
        predictions = tree.predict(self.X_test)

        expected_predictions = np.array([1.5, 5.5, 9.5]) 
        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=0,
                                       err_msg="Предсказания не соответствуют ожидаемым значениям")

    def test_criterion_mse(self):
        tree = DecisionTreeRegressor(criterion='mse', max_depth=3)
        tree.fit(self.X_train, self.y_train)
        predictions = tree.predict(self.X_test)
        
        expected_predictions = np.array([1.5, 5.5, 9.5])  # приблизительные значения
        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=0,
                                       err_msg="Предсказания с MSE критерием некорректны")

    def test_criterion_mae(self):
        tree = DecisionTreeRegressor(criterion='mae', max_depth=3)
        tree.fit(self.X_train, self.y_train)
        predictions = tree.predict(self.X_test)
        
        expected_predictions = np.array([1.5, 5.5, 9.5])  # приблизительные значения
        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=0,
                                       err_msg="Предсказания с MAE критерием некорректны")

    def test_max_depth(self):
        tree = DecisionTreeRegressor(max_depth=1)
        tree.fit(self.X_train, self.y_train)
        predictions = tree.predict(self.X_test)
        
        expected_predictions = np.array([3.1, 3.1, 8.2])
        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=1,
                                       err_msg="Глубина дерева не ограничивается max_depth")

    def test_min_samples_split(self):
        tree = DecisionTreeRegressor(min_samples_split=6)
        tree.fit(self.X_train, self.y_train)
        predictions = tree.predict(self.X_test)
        
        expected_predictions = np.array([5.5, 5.5, 5.5])
        np.testing.assert_almost_equal(predictions, expected_predictions, decimal=0,
                                       err_msg="min_samples_split не работает корректно")

    def test_boundary_cases(self):
        X_train_small = np.array([[1], [2]])
        y_train_small = np.array([1.0, 2.0])
        tree = DecisionTreeRegressor(max_depth=1)
        tree.fit(X_train_small, y_train_small)
        predictions = tree.predict(np.array([[1.5]]))
        
        expected_prediction = 1.5
        self.assertAlmostEqual(predictions[0], expected_prediction, places=1,
                               msg="Ошибка на граничном случае с маленькими данными")
