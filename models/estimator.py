from abc import ABC, abstractmethod

class Estimator(ABC):
    """
    Базовый абстрактный класс для всех эстиматоров в библиотеке.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Обучает модель на данных X и метках y.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные.
        y : array-like, shape (n_samples,)
            Метки классов или целевые переменные.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Предсказывает метки классов или значения для входных данных X.

        Параметры:
        X : array-like, shape (n_samples, n_features)
            Входные данные для предсказания.

        Возвращает:
        array-like, shape (n_samples,)
            Предсказанные метки классов или значения.
        """
        pass

    def score(self, X, y, metric):
        """
        Оценка модели с использованием заданной метрики.
        
        Параметры:
        X : array-like
            Признаки.
        y : array-like
            Истинные метки.
        metric : function
            Функция для оценки модели (например, accuracy, log_loss).
        
        Возвращает:
        float
            Результат оценки.
        """
        y_pred = self.predict(X)
        return metric(y, y_pred)
