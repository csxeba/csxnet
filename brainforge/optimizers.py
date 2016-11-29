import numpy as np


class SGD:

    def __init__(self, layer, eta=0.01, lambda1=0.0, lambda2=0.0):
        self.layer = layer
        self.eta = eta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __call__(self, m):
        eta = self.eta / m
        self.layer.weights -= self.layer.nabla_w * eta
        self.layer.biases -= self.layer.nabla_b * eta


class Momentum(SGD):

    def __init__(self, layer, eta=0.1, mu=0.9, nesterov=False):
        SGD.__init__(layer, eta)
        self.mu = mu
        self.nesterov = nesterov
        self.vW = np.zeros_like(layer.weights)
        self.vb = np.zeros_like(layer.biases)

    def __call__(self, m):
        eta = self.eta / m
        self.vW *= self.mu
        self.vb *= self.mu
        deltaW = self.layer.weights + self.vW if self.nesterov else self.layer.nabla_w
        deltab = self.layer.biases + self.vb if self.nesterov else self.layer.nabla_b
        self.vW += deltaW * eta
        self.vb += deltab * eta
        self.layer.weights -= self.vW
        self.layer.biases -= self.vb


class Adagrad(SGD):

    def __init__(self, layer, eta=0.01, epsilon=1e-8):
        SGD.__init__(self, layer, eta)
        self.epsilon = epsilon
        self.mW = np.zeros_like(layer.weights)
        self.mb = np.zeros_like(layer.biases)

    def __call__(self, m):
        eta = self.eta / m
        self.mW += self.layer.nabla_w ** 2
        self.mb += self.layer.nabla_b ** 2
        self.layer.weights -= (eta / np.sqrt(self.mW + self.epsilon)) * self.layer.nabla_w
        self.layer.biases -= (eta / np.sqrt(self.mb + self.epsilon)) * self.layer.nabla_b


class RMSprop(Adagrad):

    def __init__(self, layer, eta=0.1, decay=0.9, epsilon=1e-8):
        Adagrad.__init__(self, layer, eta, epsilon)
        self.decay = decay

    def __call__(self, m):
        eta = self.eta / m
        self.mW = self.decay * self.mW + (1 - self.decay) * self.layer.nabla_w**2
        self.mb = self.decay * self.mb + (1 - self.decay) * self.layer.nabla_b**2
        self.layer.weights -= eta * self.layer.nabla_w / (np.sqrt(self.mW) + self.epsilon)
        self.layer.biases -= eta * self.layer.nabla_b / (np.sqrt(self.mb) + self.epsilon)


class Adam(SGD):

    def __init__(self, layer, eta=0.1, decay_memory=0.9, decay_velocity=0.999, epsilon=1e-8):
        SGD.__init__(self, layer, eta)
        self.decay_memory = decay_memory
        self.decay_velocity = decay_velocity
        self.epsilon = epsilon

        self.mW = np.zeros_like(layer.weights)
        self.mb = np.zeros_like(layer.biases)
        self.vW = np.zeros_like(layer.weights)
        self.vb = np.zeros_like(layer.biases)

    def __call__(self, m):
        eta = self.eta / m
        self.mW = self.decay_memory*self.mW + (1-self.decay_memory)*self.layer.nabla_w
        self.mb = self.decay_memory*self.mb + (1-self.decay_memory)*self.layer.nabla_b
        self.vW = self.decay_velocity*self.vW + (1-self.decay_velocity)*(self.layer.nabla_w**2)
        self.vb = self.decay_velocity*self.vb + (1-self.decay_velocity)*(self.layer.nabla_b**2)

        self.layer.weights -= eta * self.mW / (np.sqrt(self.vW) + self.epsilon)
        self.layer.biases -= eta * self.mb / (np.sqrt(self.vb) + self.epsilon)


optimizer = {key.lower(): cls for key, cls in locals().items() if key != "np"}
