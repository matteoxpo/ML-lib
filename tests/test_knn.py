import unittest
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor as SklearnKNeighborsRegressor
from data.datasets import load_iris_dataset, load_boston_dataset
from models.knn import KNeighborsClassifier, KNeighborsRegressor

class TestKNeighbors(unittest.TestCase):

    def setUp(self):
        self.X_iris, self.y_iris = load_iris_dataset()
        self.X_boston, self.y_boston = load_boston_dataset()

        self.classifier = KNeighborsClassifier(n_neighbors=3)
        self.regressor = KNeighborsRegressor(n_neighbors=3)

        self.sklearn_classifier = SklearnKNeighborsClassifier(n_neighbors=3, algorithm='brute')
        self.sklearn_regressor = SklearnKNeighborsRegressor(n_neighbors=3, algorithm='brute')

        self.classifier.fit(self.X_iris.values, self.y_iris.values)
        self.sklearn_classifier.fit(self.X_iris.values, self.y_iris.values)

        self.regressor.fit(self.X_boston.values, self.y_boston.values)
        self.sklearn_regressor.fit(self.X_boston.values, self.y_boston.values)

    def test_classifier(self):
        predictions = self.classifier.predict(self.X_iris.values[:10])
        sklearn_predictions = self.sklearn_classifier.predict(self.X_iris.values[:10])

        np.testing.assert_array_equal(predictions, sklearn_predictions, "Предсказания классификатора не совпадают с предсказаниями scikit-learn.")

    def test_regressor(self):
        predictions = self.regressor.predict(self.X_boston.values[:10])
        sklearn_predictions = self.sklearn_regressor.predict(self.X_boston.values[:10])

        tolerance = 0.1
        for pred, sklearn_pred in zip(predictions, sklearn_predictions):
            self.assertAlmostEqual(pred, sklearn_pred, delta=tolerance, msg=f"Предсказание {pred} слишком далеко от предсказания scikit-learn {sklearn_pred}.")

if __name__ == '__main__':
    unittest.main()
