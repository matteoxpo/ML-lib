import numpy as np
import unittest
from data import load_boston_dataset, load_mnist_dataset
from models import DecisionTreeRegressor 
from models import DecisionTreeClassifier
from utils import mean_squared_error

class TestDecisionTreeRegressor(unittest.TestCase):
    
    def setUp(self):
        """Создание экземпляра регрессора и загрузка данных."""
        self.regressor = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
        X, y = load_boston_dataset()
        self.X = np.array(X)
        self.y = np.array(y)

    def test_fit_predict(self):
        """Тестирует, что модель может быть обучена и сделать предсказания."""
        self.regressor.fit(self.X, self.y)
        predictions = self.regressor.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), "Количество предсказанных значений не совпадает с количеством целевых значений.")

    def test_predict_shape(self):
        """Проверяет, что предсказания имеют правильную форму."""
        self.regressor.fit(self.X, self.y)
        predictions = self.regressor.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0],), "Предсказания должны иметь форму (n_samples,).")

    def test_prediction_accuracy(self):
        """Тестирует точность предсказаний модели с использованием MSE."""
        self.regressor.fit(self.X, self.y)
        predictions = self.regressor.predict(self.X)
        
        mse = mean_squared_error(predictions, self.y)
        self.assertLess(mse, 25, "MSE слишком велик; модель требует улучшения.")

    def test_prediction_with_new_data(self):
        """Тестирует предсказания для новых данных."""
        self.regressor.fit(self.X, self.y)
        
        new_data = np.array([[0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        prediction = self.regressor.predict(new_data)
        self.assertIsInstance(prediction, np.ndarray, "Предсказание должно быть массивом NumPy.")
        self.assertEqual(prediction.shape, (1,), "Предсказание для нового образца должно возвращать одно значение.")


class TestDecisionTreeClassifier(unittest.TestCase):
    
    def setUp(self):
        """Создание экземпляра классификатора и загрузка данных."""
        self.classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
        (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
        
        train_filter = (y_train == 0) | (y_train == 1)
        
        self.X_train = np.array(X_train[train_filter][:100])
        self.y_train = np.array(y_train[train_filter][:100])

        test_filter = (y_test == 0) | (y_test == 1)
        
        self.X_test = np.array(X_test[test_filter][100:120])
        self.y_test = np.array(y_test[test_filter][100:120])
        
        
        

    def test_fit_predict(self):
        """Тестирует, что модель может быть обучена и сделать предсказания."""
        self.classifier.fit(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test)
        
        self.assertEqual(predictions.shape, self.y_test.shape, "Форма предсказаний не совпадает с формой тестовых данных.")
    
    def test_accuracy(self):
        """Тестирует, что точность классификатора превышает хотя бы 50%."""
        self.classifier.fit(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test)
        
        accuracy = np.mean(predictions == self.y_test)
        self.assertGreaterEqual(accuracy, 0.5, "Точность классификатора должна быть хотя бы 50%.")