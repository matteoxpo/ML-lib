import numpy as np
from estimator import Estimator
from decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingClassifier(Estimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        y_pred = np.full(y.shape, np.mean(y))

        for _ in range(self.n_estimators):
            diff = y - y_pred

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, diff)

            y_pred += self.learning_rate * tree.predict(X)

            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
