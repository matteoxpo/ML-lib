import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, batch_size=1, shuffle=True):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.velocity = None

    def iterate(self, X, y, model):
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        
        indices = np.arange(X.shape[0])
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, X.shape[0], self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            
            gradients = model.compute_gradients(X_batch, y_batch)
            model.params = self._update(model.params, gradients)
    
    def _update(self, params, gradients):
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in params]
        
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * gradients[i]
            params[i] += self.velocity[i]
        
        return params
