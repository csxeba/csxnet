"""
Feed Forward Neural Network with Layer level simulation.
Faster than Neuron level implementation.

Copyright (c) 2015 Csaba Gór
contact:  csabagor@gmail.com

This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the
    Free Software Foundation.

    This program is distributed but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import random
import math

from .NNModel import FFNN
from ..Utility.activations import *
from ..Utility.cost import Xent


class FFNeuralBrain:
    """Neural-level simulation of the network.

    Significantly slower than LayerBrain, because it instanciates every neuron and doesn't use numpy.
    It will also be the superclass for the Recurrent Network (or the RLayer)"""

    def __init__(self, rate, layout, activation="sigmoid"):

        print("Warning! Not working, please do not use!")

        self.layout = layout
        self.rate = rate

        self.activation, self.activation_p = functions[activation]

        self.hiddens = []

        for i, n in enumerate(self.layout[1:]):
            self.hiddens.append([FFNeuron(inputs=self.layout[i], position=[i + 1, j])
                                 for j in range(n)])

        self.inputL = [InputNeuron(layout[0]) for _ in range(self.layout[0])]
        self.outputL = self.hiddens[-1]
        self.layers = [self.inputL] + self.hiddens

        self.stimuli = None
        self.error = 0
        self.age = 0

    def think(self, stimuli):
        return np.array([self._feedforward(stimulus) for stimulus in stimuli])

    def learn(self, table):
        for lesson, target in zip(table[0], table[1]):
            self._feedforward(stimulus=lesson)
            self._backpropagation(targets=target)
            self._weight_update()
            self.error = sum([neu.error for neu in self.hiddens[-1]])

    def shuffle(self, zero):
        for i in self.hiddens:
            for j in i:
                if zero:
                    j.weights = [0 for _ in range(len(j.weights))]
                else:
                    j.weights = [random.gauss(0, 1) for _ in range(len(j.weights))]

    def _feedforward(self, stimulus):
        stimulus = list(stimulus)
        for layer in self.layers:
            stimulus = [neuron.fire(stimulus) for neuron in layer]

        return stimulus

    def _backpropagation(self, targets):
        for target, neu in zip(targets, self.outputL):
            neu.error = self.activation_p(neu.excitation) * (target - neu.output)

        for layer in self.hiddens[-2::-1]:
            for neu in layer:
                neu.error = self.activation_p(neu.excitation) * \
                            sum([neu1.error * neu1.weights[neu.position[1]]
                                 for neu1 in self.layers[neu.position[0] + 1]])

    def _weight_update(self):
        for layer, prevLayer in zip(self.layers[-1:0:-1],
                                    self.layers[-2::-1]):
            for neu in layer:
                neu.weights = [weight + prevLayer[index].output * neu.error * self.rate
                               for index, weight in enumerate(neu.weights)]
                neu.bias += neu.error

    def _sigmoid(self, z):
        return 1 / (1 + math.exp(-z))


class FFLayerBrain(FFNN):
    """This was left here for compatibility purposes"""
    def __init__(self, hiddens, data, eta, lmbd1=0.0, lmbd2=0.0, mu=0.0,
                 cost=Xent, activation=Sigmoid):
        FFNN.__init__(self, hiddens, data, eta, lmbd1, lmbd2, mu, cost, activation)
