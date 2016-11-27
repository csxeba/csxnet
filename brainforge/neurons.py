import random
import math


class Neuron:
    def __init__(self, inputs, activation):
        self.weights = [random.normalvariate(0, 1) / math.sqrt(inputs) for _ in range(inputs)]
        self.bias = 0.0
        self.nabla_w = [0.0 for _ in range(inputs)]
        self.nabla_b = 0.0
        self.output = 0.0
        self.inputs = [0.0 for _ in range(inputs)]

        self.activation = activation

    def fire(self, stimuli):
        self.inputs = stimuli
        z = sum([inpt * wght for inpt, wght in zip(stimuli, self.weights)]) + self.bias
        self.output = self.activation(z)
        return self.output

    def backpropagate(self, error):
        delta = [self.activation.derivative(e) for e in error]
        self.nabla_b = sum(delta)
        self.nabla_w = [sum([d * inpt for d in delta]) for inpt in self.inputs]
        return [sum([w * d for w in self.weights]) for d in error]


class Recurrent(Neuron):
    def __init__(self, inputs, activation, return_seq=False):
        Neuron.__init__(self, inputs, activation)
        self.time = 0
        self.return_seq = return_seq
        self.outputs = []
        self.Z = [0.0 for _ in range(inputs + 1)]

    def fire(self, stimuli):
        self.inputs = stimuli
        self.time = len(stimuli)
        h = 0.0
        outputs = []
        for t in range(self.time):
            z = sum([inpt * wght for inpt, wght in zip(stimuli[t] + [h], self.weights)])
            outputs.append(self.activation(z))
        if self.return_seq:
            self.output = outputs
        else:
            self.output = outputs[-1]
        return self.output

    def backpropagate(self, error):
        delta = 0.0
        self.nabla_w = [0.0 for _ in range(len(self.nabla_w))]
        self.nabla_b = 0.0
        # deltaX = [[0.0 for _ in range(len(self.inputs[0]))] for _ in range(len(self.inputs))]
        for t in range(self.time, -1, -1):
            delta += error
            delta = self.activation.derivative(delta)
            self.nabla_b += sum(delta)
            self.nabla_w = [nw + (sum(d * inpt for d in delta) for inpt in self.inputs) for nw in self.nabla_w]
        raise NotImplemented


