# data/datasets.py

import pandas as pd

def load_iris_dataset():
    """
    Загрузить датасет Iris (Ирисы).
    Возвращает pandas DataFrame c фичами и pandas Series с целевой переменной.
    """
    iris = pd.read_csv('sets/Iris.csv')
    X = iris.drop('Species', axis=1)
    y = iris['Species']
    return X, y

def load_boston_dataset():
    """
    Загрузить датасет Boston Housing (Цены на жилье в Бостоне).
    Возвращает pandas DataFrame c фичами и pandas Series с целевой переменной.
    """
    data = pd.read_csv('sets/BostonHousing.csv')
    X = data.drop('medv', axis=1)
    y = data['medv']
    return X, y

def load_mnist_dataset():
    """
    Загрузить датасет MNIST (рукописные цифры).
    Возвращает тренировочный и тестовый наборы данных.
    """
    data = pd.read_csv('sets/mnist_train.csv')
    y_train = data.iloc[:, 0]
    X_train = data.iloc[:, 1:]
   
    data = pd.read_csv('sets/mnist_test.csv')
    y_test = data.iloc[:, 0]
    X_test = data.iloc[:, 1:]

    return (X_train, y_train), (X_test, y_test)