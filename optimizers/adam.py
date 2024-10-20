import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=1, shuffle=True):
        """
        :param learning_rate: скорость обучения
        :param beta1: коэффициент для первого момента (обычно 0.9)
        :param beta2: коэффициент для второго момента (обычно 0.999)
        :param epsilon: небольшое значение для предотвращения деления на ноль
        :param batch_size: размер мини-батча
        :param shuffle: перемешивать данные перед итерацией
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.m = None  # Первый момент
        self.v = None  # Второй момент
        self.t = 0  # Итерация

    def iterate(self, X, y, params):
        """
        :param X: входные данные
        :param y: целевые значения
        :param params: параметры модели
        """
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        
        indices = np.arange(X.shape[0])
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, X.shape[0], self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            
            gradients = self._compute_dummy_gradients(X_batch, y_batch, params)

            self._update(params, gradients)
    
    def _compute_dummy_gradients(self, X, y, params):
        """
        Фиктивная функция для вычисления градиентов.
        Здесь мы просто возвращаем случайные градиенты.
        """
        return [np.random.randn(*param.shape) for param in params]

    def _update(self, params, gradients):
        if self.m is None:
            self.m = [np.zeros_like(param) for param in params]
        if self.v is None:
            self.v = [np.zeros_like(param) for param in params]

        self.t += 1
        
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gradients[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)