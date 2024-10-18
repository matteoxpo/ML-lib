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
│   ├── datasets.py             #DONE
│   ├── label_encoder.py        #DONE
│   └── preprocessing.py        #DONE
├── models/
│   ├── ensemble/
│   │   ├── __init__.py                        
│   │   ├── bagging.py                         
│   │   ├── blending.py                        
│   │   ├── boosting.py                        
│   │   ├── gradient_boosting_classifier.py        #DONE
│   │   ├── gradient_boosting_regressor.py         #DONE
│   │   └── stacking.py                        
│   ├── unsupervised/
│   │   ├── __init__.py    
│   │   ├── kemas.py       
│   │   └── pca.py         
│   ├── __init__.py            
│   ├── decision_tree.py           #DONE
│   ├── estimator.py               #DONE
│   ├── gradient_descent.py        #DONE
│   ├── grid_searcher.py           #DONE
│   ├── knn.py                     #DONE
│   ├── linear.py                  #DONE
│   └── sgd.py                     #DONE
├── optimizers/
│   ├── __init__.py            
│   ├── adam.py                
│   ├── gradient_descent.py        #DONE
│   └── sgd.py                     #DONE
├── readme/
│   └── hierarchy.py        #DONE
├── recommender/
│   ├── __init__.py                   
│   ├── collabarative_filtering.py    
│   └── matrix_factorization.py       
├── tests/
│   ├── test_datasets.py        #DONE
│   ├── test_knn.py             #DONE
│   ├── test_linear.py          #DONE
│   └── test_metrics.py         #DONE
└── utils/
    ├── __init__.py    
    ├── metrics.py         #DONE
    └── utils.py           #DONE
```
![GitHub](https://img.shields.io/badge/github-matteoxpo-orange)
![Telegram](https://img.shields.io/badge/telegram-xpomin-blue)