import random
import math


class Neuron:
    def __init__(self, inputs, position, activation):
        self.position = position
        self.weights = [random.normalvariate(0, 1) / math.sqrt(inputs) for _ in range(inputs)]
        self.bias = random.normalvariate(0, 1)
        self.excitation = 0
        self.output = 0
        self.error = 0

        self.activation = activation

    def fire(self, stimuli):
        self.excitation = sum([inpt * wght for inpt, wght in zip(stimuli, self.weights)]) + self.bias
        self.output = self.activation(self.excitation)
        return self.output


class InputNeuron(Neuron):
    def __init__(self, inputs):
        Neuron.__init__(self, inputs=inputs, position=0, activation="linear")
        self.weights = [1] * inputs
