import numpy as np
import unittest
from models.ensemble import GradientBoostingDecisionTreeClassifier
from models import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification
from utils import accuracy

from data import load_mnist_dataset

class TestGradientBoostingDecisionTreeClassifier(unittest.TestCase):

    def setUp(self):
        """Настройка тестовых данных."""
        self.X, self.y = make_classification(
            n_samples=500,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42,
            class_sep=1.5,
            flip_y=0.06,
            weights=[0.6, 0.4]
        )
        
        self.model = GradientBoostingDecisionTreeClassifier(n_estimators=10, g_probabilities=True, learning_rate=0.2, max_depth=10, min_samples_split=4)
        
    def test_predict(self):
        """Проверка предсказаний модели."""
        
        self.model.fit(np.copy(self.X), np.copy(self.y))
        predictions = self.model.predict(self.X)
        
        predictions_binary = (predictions >= 0.5).astype(int)
        
        accuracy_score = accuracy(self.y, predictions_binary)
        self.assertGreater(accuracy_score, 0.5)

