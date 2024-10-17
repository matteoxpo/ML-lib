import numpy as np
import unittest
from models import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier as DecTree

#    sklern_tree =DecTree(
#             criterion='gini',              
#             max_depth=2,                
#             min_samples_split=2,           
#             min_samples_leaf=1,            
#             max_features=None,             
#             random_state=42              
#         )
        
#         sklern_tree.fit(self.X_train, self.y_train)
#         print(sklern_tree.predict(self.X_test))
#         print(predictions)

class TestDecisionTreeClassifier(unittest.TestCase):
    
    def setUp(self):
        # Создаем тестовый набор данных
        self.X_train = np.array([[0, 0], 
                                 [1, 1], 
                                 [0, 1], 
                                 [1, 0]])
        self.y_train = np.array([0, 
                                 1, 
                                 1, 
                                 0])
        self.X_test = np.array([[0, 0], 
                                [1, 1], 
                                [0, 1], 
                                [1, 0]])

    def test_fit_and_predict(self):
        # Проверяем, что классификатор может обучаться и делать предсказания
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)
        expected_predictions = np.array([0, 1, 1, 0]) 
        np.testing.assert_array_equal(predictions, expected_predictions)

    # def test_predict_with_single_class(self):
    #     # Проверяем, что предсказания работают, когда есть только один класс
    #     clf = DecisionTreeClassifier(max_depth=2)
    #     clf.fit(self.X_train, self.y_train)
    #     single_class_X = np.array([[0, 0], [0, 0]])  # Все примеры одного класса
    #     single_class_y = np.array([0, 0])  # Ожидаемый класс
    #     clf.fit(single_class_X, single_class_y)
    #     predictions = clf.predict(self.X_test)
    #     np.testing.assert_array_equal(predictions, np.array([0, 0, 0, 0]))

    # def test_most_common_label(self):
    #     # Проверяем, что метод _most_common_label возвращает правильную метку
    #     clf = DecisionTreeClassifier()
    #     common_label = clf._most_common_label(np.array([1, 1, 0, 1, 0, 0, 0]))
    #     self.assertEqual(common_label, 1)  # 1 наиболее часто встречается

    # def test_calculate_impurity_gini(self):
    #     # Проверяем, что gini impurity вычисляется правильно
    #     clf = DecisionTreeClassifier(criterion='gini')
    #     impurity = clf._calculate_impurity(np.array([1, 1, 0, 0, 0, 1]))
    #     expected_impurity = 0.5  # Расчет: 1/2 * (3/6 * (1 - (3/3)^2) + 3/6 * (1 - (3/3)^2))
    #     self.assertAlmostEqual(impurity, expected_impurity)

    # def test_calculate_impurity_entropy(self):
    #     # Проверяем, что entropy impurity вычисляется правильно
    #     clf = DecisionTreeClassifier(criterion='entropy')
    #     impurity = clf._calculate_impurity(np.array([1, 1, 0, 0, 0, 1]))
    #     expected_impurity = 0.9182958340544896  # Расчет на основе вероятностей
    #     self.assertAlmostEqual(impurity, expected_impurity)

    # def test_best_split(self):
    #     # Проверяем, что наилучшее разбиение вычисляется правильно
    #     clf = DecisionTreeClassifier()
    #     index, threshold, error = clf._best_split(self.X_train, self.y_train)
    #     expected_feature_index = 0
    #     expected_threshold = 0.5
    #     self.assertEqual(index, expected_feature_index)
    #     self.assertEqual(threshold, expected_threshold)

if __name__ == '__main__':
    unittest.main()
