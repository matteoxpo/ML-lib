import unittest
import numpy as np
from models.ensemble import BaggingDecisionTreeClassifier, BaggingDecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from utils import accuracy, mean_squared_error

class TestBaggingModels(unittest.TestCase):
    
    def test_bagging_classifier(self):
        """Тест для бэггинг-классификатора"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = BaggingDecisionTreeClassifier(n_estimators=10, max_depth=6, g_threshold=0.5)
        
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        accuracy_score = accuracy(y, y_pred)
        
        self.assertGreater(accuracy_score, 0.6, "Accuracy is lower than expected for the classifier.")
    
    def test_bagging_regressor(self):
        """Тест для бэггинг-регрессора"""
        X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
        
        model = BaggingDecisionTreeRegressor(n_estimators=220, max_depth=10, min_samples_split=2)
        
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        
        self.assertLess(mse, 250, "MSE is higher than expected for the regressor.")
    
    def test_classifier_with_probabilities(self):
        """Тест для классификатора с включенными вероятностями"""
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        model = BaggingDecisionTreeClassifier(n_estimators=10, max_depth=3, g_probabilities=True)
        
        model.fit(X, y)
        
        y_prob = model.predict(X)
        
        self.assertTrue(np.all((y_prob >= 0) & (y_prob <= 1)), "Probabilities should be in range [0, 1].")


class TestRandomForest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        self.y_class = np.array([0, 0, 1, 1, 1])
        self.y_reg = np.array([1, 2, 3, 4, 5]) 

    def test_random_forest_classifier(self):
        model = RandomForestClassifier(n_estimators=10, max_depth=10, g_probabilities=True, max_features=1)
        model.fit(self.X, self.y_class)
        
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y_class))

        probs = model._predict_sample(self.X[0])
        self.assertGreaterEqual(probs, 0)
        self.assertLessEqual(probs, 1)

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(n_estimators=10, max_depth=10, max_features=1)
        model.fit(self.X, self.y_reg)
        
        preds = model.predict(self.X)
        self.assertEqual(len(preds), len(self.y_reg))
        
        self.assertTrue(np.all(preds >= 1))