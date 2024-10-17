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
├── data/
│   ├── sets/
│   │   ├── BostonHousing.csv
│   │   ├── Iris.csv
│   │   ├── mnist_test.csv
│   │   └── mnist_train.csv
│   ├── __init__.py
│   ├── datasets.py
│   ├── label_encoder.py
│   └── preprocessing.py
├── models/
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── bagging.py
│   │   ├── blending.py
│   │   ├── boosting.py
│   │   └── stacking.py
│   ├── unsupervised/
│   │   ├── __init__.py
│   │   ├── kemas.py
│   │   └── pca.py
│   ├── __init__.py
│   ├── decision_tree_classifier.py
│   ├── decision_tree_node.py
│   ├── decision_tree_regressor.py
│   ├── estimator.py
│   ├── gradient_boosting.py
│   ├── grid_searcher.py
│   ├── knn.py
│   ├── linear_classifier.py
│   ├── linear_regressor.py
│   ├── sgd_classifier.py
│   └── sgd_regressor.py
├── optimizers/
│   ├── __init__.py
│   ├── adam.py
│   ├── gradient_descent.py
│   └── sgd.py
├── recommender/
│   ├── __init__.py
│   ├── collabarative_filtering.py
│   └── matrix_factorization.py
├── tests/
│   ├── test_datasets.py
│   ├── test_decision_tree_classifier.py
│   ├── test_decision_tree_regressor.py
│   ├── test_knn.py
│   ├── test_linear_regresion.py
│   └── test_metrics.py
└── utils/
    ├── __init__.py
    ├── metrics.py
    └── utils.py
```
![GitHub](https://img.shields.io/badge/github-matteoxpo-orange)
![Telegram](https://img.shields.io/badge/telegram-xpomin-blue)