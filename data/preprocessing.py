# data/preprocessing.py

import numpy as np

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделить данные на тренировочные и тестовые наборы.
    
    Параметры:
    - X: Признаки (numpy array).
    - y: Метки (numpy array).
    - test_size: Доля данных для тестирования (по умолчанию 0.2).
    - random_state: Контролирует перемешивание данных перед разделением.
    
    Возвращает:
    - X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    test_size = int(len(X) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def standardize_data(X_train, X_test=None):
    """
    Стандартизировать данные (привести к нулевому среднему и единичной дисперсии).
    
    Параметры:
    - X_train: Тренировочные признаки (numpy array).
    - X_test: Тестовые признаки (опционально).
    
    Возвращает стандартизированные данные.
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    X_train_scaled = (X_train - mean) / std
    
    if X_test is not None:
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled
    
    return X_train_scaled

def normalize_data(X_train, X_test=None):
    """
    Нормализовать данные (привести значения к диапазону [0, 1]).
    
    Параметры:
    - X_train: Тренировочные признаки (numpy array).
    - X_test: Тестовые признаки (опционально).
    
    Возвращает нормализованные данные.
    """
    X_min = np.min(X_train, axis=0)
    X_max = np.max(X_train, axis=0)
    
    X_train_normalized = (X_train - X_min) / (X_max - X_min)
    
    if X_test is not None:
        X_test_normalized = (X_test - X_min) / (X_max - X_min)
        return X_train_normalized, X_test_normalized
    
    return X_train_normalized

def scale_data(X, method='min-max'):
    """
    Масштабирует данные с использованием выбранного метода.
    
    Параметры:
    X : array-like, shape (n_samples, n_features)
        Входные данные для масштабирования.
    method : str, optional (default='min-max')
        Метод масштабирования: 'min-max' или 'z-score'.
    
    Возвращает:
    numpy.ndarray
        Масштабированные данные.
    """
    X = np.array(X)
    
    if method == 'min-max':
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        scaled_X = (X - X_min) / (X_max - X_min)
    
    elif method == 'z-score':
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        scaled_X = (X - X_mean) / X_std
    
    else:
        raise ValueError("Метод должен быть 'min-max' или 'z-score'.")
    
    return scaled_X