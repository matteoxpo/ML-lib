import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0
        
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


class GridSearchCV:
    """
    Поиск по сетке для выбора оптимальных гиперпараметров модели.
    
    Параметры:
    estimator : object
        Объект модели.
    param_grid : dict
        Словарь с параметрами и значениями для перебора.
    scoring : function
        Функция для оценки производительности модели.
    cv : int
        Количество фолдов для кросс-валидации.
    """
    
    def __init__(self, estimator, param_grid, scoring, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    def fit(self, X, y):
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[param] for param in param_names]
        
        for param_combination in self._get_combinations(param_values):
            for i, param_name in enumerate(param_names):
                setattr(self.estimator, param_name, param_combination[i])
            
            scores = []
            kf = KFold(n_splits=self.cv)
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                self.estimator.fit(X_train, y_train)
                y_pred = self.estimator.predict(X_val)
                score = self.scoring(y_val, y_pred)
                scores.append(score)
            
            mean_score = np.mean(scores)
            
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = param_combination
                self.best_estimator_ = self.estimator
        
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    def score(self, X, y):
        return self.scoring(y, self.predict(X))
    
    def _get_combinations(self, param_values):
        from itertools import product
        return list(product(*param_values))
