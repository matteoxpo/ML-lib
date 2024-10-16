import unittest

import sklearn.linear_model
from data.datasets import load_boston_dataset
from models.linear_regressor import LinearRegressor 

class TestLinearRegressor(unittest.TestCase):

    def setUp(self):
        self.X_boston, self.y_boston = load_boston_dataset()
        
        self.model_boston = LinearRegressor(fit_intercept=True)
        self.model_boston.fit(self.X_boston.values, self.y_boston)

    def test_fit(self):
        expected_shape_boston = self.X_boston.shape[1] + 1
        self.assertEqual(len(self.model_boston.weights), expected_shape_boston)

    def test_predict(self):
        predictions = self.model_boston.predict(self.X_boston.values)
        for i in range(len(predictions)):
            self.assertFalse(predictions[i] - self.y_boston[i] > 20) 

    def test_shape(self):
        predictions = self.model_boston.predict(self.X_boston.values)
        self.assertEqual(predictions.shape, self.y_boston.shape)

if __name__ == '__main__':
    unittest.main()
