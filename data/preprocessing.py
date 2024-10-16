# data/preprocessing.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделить данные на тренировочные и тестовые наборы.
    
    Параметры:
    - X: Признаки (DataFrame или numpy array).
    - y: Метки (Series или numpy array).
    - test_size: Доля данных для тестирования (по умолчанию 0.2).
    - random_state: Контролирует перемешивание данных перед разделением.
    
    Возвращает:
    - X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test=None):
    """
    Стандартизировать данные (привести к нулевому среднему и единичной дисперсии).
    
    Параметры:
    - X_train: Тренировочные признаки (DataFrame или numpy array).
    - X_test: Тестовые признаки (опционально).
    
    Возвращает стандартизированные данные и объект StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

def normalize_data(X_train, X_test=None):
    """
    Нормализовать данные (привести значения к диапазону [0, 1]).
    
    Параметры:
    - X_train: Тренировочные признаки (DataFrame или numpy array).
    - X_test: Тестовые признаки (опционально).
    
    Возвращает нормализованные данные и объект MinMaxScaler.
    """
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_normalized = scaler.transform(X_test)
        return X_train_normalized, X_test_normalized, scaler
    
    return X_train_normalized, scaler
