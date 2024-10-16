# ML Library

![Python](https://img.shields.io/badge/python-3.6%2B-blue)

## Описание

Это библиотека для машинного обучения, разработанная с целью обучения и глубокого понимания методов машинного обучения. Библиотека имеет функционал, схожий с популярной библиотекой [scikit-learn](https://scikit-learn.org/).

## Цели

- **Обучение**: Библиотека создаётся для того, чтобы автор смог самостоятельно написать и понять алгоритмы машинного обучения.
- **Понимание**: Основное внимание уделяется углубленному пониманию того, как работают различные методы, чтобы в будущем можно было применять их более эффективно.

## Структура проекта
```bash
ml-lib/
│  
├── data/                            # Для хранения и обработки данных
│   ├── __init__.py
│   ├── datasets.py                  # Тестовые датасеты
│   ├── preprocessing.py             # Методы предварительной обработки данных
│   └── sets/
│       ├── BostonHousing.csv 
│       ├── Iris.csv
│       ├── mnist_test.csv
│       └── mnist_train.csv
│
├── models/                          # Основные модели
│   ├── __init__.py
│   ├── grid_search.py               # Перебор параметров модели 
│   ├── knn.py                       # k-Nearest Neighbors
│   ├── linear_regression.py         # Линейная регрессия
│   ├── decision_tree.py             # Деревья решений
│   ├── gradient_boosting.py         # Градиентный бустинг
│   ├── linear_classifier.py         # Линейный классификатор
│   ├── unsupervised/                # Обучение без учителя
│   │   ├── __init__.py
│   │   ├── kmeans.py                # k-Means Clustering
│   │   └── pca.py                   # PCA (Principal Component Analysis)
│   └── ensemble/                    # Ансамблевые методы
│       ├── __init__.py
│       ├── bagging.py               # Бэггинг
│       ├── boosting.py              # Бустинг
│       ├── blending.py              # Блендинг
│       └── stacking.py              # Стэкинг
│
├── optimizers/                      # Оптимизаторы (Градиентный спуск и др.)
│   ├── __init__.py
│   ├── gradient_descent.py          # Градиентный спуск
│   ├── adam.py                      # Оптимизатор Adam
│   └── sgd.py                       # Стохастический градиентный спуск (SGD)
│
├── recommenders/                    # Рекомендательные системы
│   ├── __init__.py
│   ├── collaborative_filtering.py   # Коллаборативная фильтрация
│   └── matrix_factorization.py      # Факторизация матриц
│
├── utils/                           # Вспомогательные функции
│   ├── __init__.py
│   ├── metrics.py                   # Метрики качества (MSE, Accuracy и т.д.)
│   └── utils.py                     # Общие утилиты (например, функции для отладки)
│
└── tests/                           # Тесты для каждой модели
    ├── test_knn.py
    ├── # ...
    ├── # other test files
    ├── # ...
    └── test_linear_regression.py
```
![GitHub](https://img.shields.io/badge/github-matteoxpo-orange)
![Telegram](https://img.shields.io/badge/telegram-xpomin-blue)