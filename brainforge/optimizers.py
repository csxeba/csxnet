class SGD:

    def __init__(self, eta=0.01, mu=0.0):
        self.eta = eta
        self.mu = mu

    def __call__(self, layer, m):
        eta = self.eta / m
        if self.mu:
            layer.velocity *= self.mu
            layer.velocity += layer.nabla_w * eta
            layer.weights -= layer.velocity
        else:
            layer.weights -= layer.nabla_w * eta

        layer.biases -= layer.nabla_b * eta
