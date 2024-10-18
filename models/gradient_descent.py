from .sgd import SGDClassifier, SGDRegressor

class GradientDescentClassifier(SGDClassifier):
    def __init__(self, 
                 learning_rate=0.01, n_iterations=1000, 
                 fit_intercept=True, 
                 shuffle=True, 
                 probabilities = False, threshold = 0.5, 
                 tol=1e-4, early_stopping=True,  n_iter_no_change=5):
        super.__init__(
                 learning_rate=learning_rate, n_iterations=n_iterations, 
                 fit_intercept=fit_intercept, 
                 batch_size=None, shuffle=shuffle, 
                 probabilities = probabilities, threshold = threshold, 
                 tol=tol, early_stopping=early_stopping,  n_iter_no_change=n_iter_no_change)
        
class GradientDescentRegressor(SGDRegressor):
    def __init__(self, 
                 learning_rate=0.01, n_iterations=1000, 
                 fit_intercept=True, 
                 shuffle=True, 
                 tol=1e-4, early_stopping=True,  n_iter_no_change=5):
        super.__init__(
                 learning_rate=learning_rate, n_iterations=n_iterations, 
                 fit_intercept=fit_intercept, 
                 batch_size=None, shuffle=shuffle, 
                 tol=tol, early_stopping=early_stopping,  n_iter_no_change=n_iter_no_change)