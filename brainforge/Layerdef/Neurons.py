import random

from ..Utility.activations import *


class FFNeuron:
    def __init__(self, inputs, position, activation=Sigmoid):
        self.position = position
        self.weights = [random.normalvariate(0, 1) for _ in range(inputs)]
        self.bias = random.normalvariate(0, 1)
        self.excitation = 0
        self.output = 0
        self.error = 0

        self.activation = Sigmoid

    def fire(self, stimuli):
        self.excitation = sum([x * y for x, y in zip(stimuli, self.weights)]) + self.bias
        self.output = self.activation(self.excitation)
        return self.output


class InputNeuron(FFNeuron):
    def __init__(self, inputs):
        FFNeuron.__init__(self, inputs=inputs, position=0, activation="linear")
        self.weights = [1] * inputs
