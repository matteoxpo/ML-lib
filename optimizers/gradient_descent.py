from sgd import SGD

class GradientDescent(SGD):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate=learning_rate, momentum=0.0, batch_size=None, shuffle=False)