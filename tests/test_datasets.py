import unittest
import pandas as pd
from data.datasets import load_iris_dataset, load_boston_dataset, load_mnist_dataset

class TestDatasets(unittest.TestCase):
    
    def test_load_iris_dataset(self):
        X, y = load_iris_dataset()
        
        # Проверяем, что X и y являются pandas DataFrame и Series соответственно
        self.assertIsInstance(X, pd.DataFrame, "X должен быть DataFrame")
        self.assertIsInstance(y, pd.Series, "y должен быть Series")
        
        # Проверяем, что в X есть 4 признака (как в датасете Iris)
        self.assertEqual(X.shape[1], 4, "Датасет Iris должен содержать 4 признака")
        
        # Проверяем, что уникальные значения y соответствуют ожидаемым классам
        expected_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.assertTrue(set(expected_classes).issubset(set(y.unique())),
                        "В y должны быть классы: {}".format(expected_classes))

    def test_load_boston_dataset(self):
        X, y = load_boston_dataset()
        
        # Проверяем, что X и y являются pandas DataFrame и Series соответственно
        self.assertIsInstance(X, pd.DataFrame, "X должен быть DataFrame")
        self.assertIsInstance(y, pd.Series, "y должен быть Series")
        
        # Проверяем, что в X есть 13 признаков (как в датасете Boston Housing)
        self.assertEqual(X.shape[1], 13, "Датасет Boston Housing должен содержать 13 признаков")

    def test_load_mnist_dataset(self):
        (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
        
        # Проверяем, что X_train и y_train являются pandas DataFrame и Series соответственно
        self.assertIsInstance(X_train, pd.DataFrame, "X_train должен быть DataFrame")
        self.assertIsInstance(y_train, pd.Series, "y_train должен быть Series")
        
        # Проверяем, что X_train и X_test имеют ожидаемые размеры
        self.assertEqual(X_train.shape[1], 784, "Датасет MNIST должен содержать 784 признака (28x28 пикселей)")
        self.assertEqual(X_test.shape[1], 784, "Датасет MNIST должен содержать 784 признака (28x28 пикселей)")

if __name__ == "__main__":
    unittest.main()
