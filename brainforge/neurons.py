import random
import math


class Neuron:
    def __init__(self, inputs, position, activation):
        self.position = position
        self.weights = [random.normalvariate(0, 1) / math.sqrt(inputs) for _ in range(inputs)]
        self.gradients = [0.0 for _ in range(inputs)]
        self.bias = 0
        self.output = 0
        self.error = 0
        self.inputs = [0.0 for _ in range(inputs)]

        self.activation = activation

    def fire(self, stimuli):
        self.inputs = stimuli
        z = sum([inpt * wght for inpt, wght in zip(stimuli, self.weights)]) + self.bias
        self.output = self.activation(z)
        return self.output

    def backpropagation(self):
        self.gradients = [sum([err * inpt for err in self.error]) for inpt in self.inputs]
        return [sum([w * err for w in self.weights]) for err in self.error]

    def weight_update(self):
        self.weights = [w - gr for w, gr in zip(self.weights, self.gradients)]

    def receive_error(self, errors):
        self.error = self.activation.derivative(errors)

