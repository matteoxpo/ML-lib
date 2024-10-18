import unittest
import numpy as np

from models import LinearRegressor, LinearClassifier  

class TestLinearModels(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y_train_regression = np.array([3, 5, 7, 9])  
        self.y_train_classification = np.array([0, 0, 1, 1])

    def test_linear_regressor_fit_and_predict(self):
        model = LinearRegressor(fit_intercept=True)
        model.fit(self.X_train, self.y_train_regression)
        predictions = model.predict(self.X_train)

        self.assertEqual(predictions.shape, self.y_train_regression.shape)

        np.testing.assert_almost_equal(predictions, self.y_train_regression, decimal=1)

    def test_linear_classifier_fit_and_predict(self):
        model = LinearClassifier(fit_intercept=True)
        model.fit(self.X_train, self.y_train_classification)
        predictions = model.predict(self.X_train)

        self.assertEqual(predictions.shape, self.y_train_classification.shape)

        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_linear_regressor_no_intercept(self):
        model = LinearRegressor(fit_intercept=False)
        model.fit(self.X_train, self.y_train_regression)
        predictions = model.predict(self.X_train)

        self.assertEqual(predictions.shape, self.y_train_regression.shape)

        np.testing.assert_almost_equal(predictions, self.y_train_regression, decimal=1)

    def test_linear_classifier_no_intercept(self):
        model = LinearClassifier(fit_intercept=False)
        model.fit(self.X_train, self.y_train_classification)
        predictions = model.predict(self.X_train)

        self.assertEqual(predictions.shape, self.y_train_classification.shape)

        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_fit_incompatible_shape(self):
        model = LinearRegressor(fit_intercept=True)

        with self.assertRaises(ValueError):
            model.fit(self.X_train, np.array([1, 2]))  