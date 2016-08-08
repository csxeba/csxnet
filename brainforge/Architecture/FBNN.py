"""This is a FFNN with some feedback capabilities implemented."""

from brainforge.activations import Sigmoid
from .FFNN import *
from ..Layerdef.Neurons import Neuron

sigmoid = Sigmoid()

RATE = 1
FEEDBACKERS_PERCENT = 25


class RNN(FFNeuralBrain):
    def __init__(self, rate, *layout):
        FFNeuralBrain.__init__(self, rate, *layout)
        if len(self.layout) < 4:
            raise RuntimeError("At least 2 hidden layers\
                                are required for recurrent networking!")

        n_feedbackers = int(len(self.hiddens[-1]) * (FEEDBACKERS_PERCENT / 100))
        self.feedbackers = self.hiddens[-1][:n_feedbackers]

        self.hiddens[0] = [Neuron(inputs=layout[0] + n_feedbackers,
                                  position=(0, i), activation=sigmoid)
                           for i in range(layout[1])]
        self.architecture = "RNN"

    def think(self, stimuli):
        stimuli = list(stimuli) + [neuron.output for neuron in self.feedbackers]
        return FFNeuralBrain.think(self, stimuli)

    def _backpropagation(self, targets):
        # output error
        for i, neu in enumerate(self.hiddens[-1]):
            target = targets[i]
            neu.error = neu.output - target
        # hidden error (backpropagation)
        for layer in self.hiddens[-1:0:-1]:
            for neu in layer:
                neu.error = sum([neu1.error * neu1.weights[neu.position[1]]
                                 for neu1 in self.hiddens[neu.position[0] + 1]])
        # error of the first hidden layer (backpropagation + feedback error)
        for neu in self.hiddens[0]:
            next_layer_errors = [neu1.error * neu1.weights[neu.position[1]]
                                 for neu1 in self.hiddens[neu.position[0] + 1]]

            feedback_errors = []
            for feedbacker in self.feedbackers:
                coef = feedbacker.weights[0]
                for weight in feedbacker.weights[1:]:
                    coef = coef * weight
                feedback_errors.append(feedbacker.error * coef)

            neu.error = sigmoid(sum(next_layer_errors + feedback_errors))

        # weight update of all the layers except the fed-back layer
        for layer in self.hiddens[-1:0:-1]:
            for neu in layer:
                neu.weights = \
                    [weight +
                     (self.hiddens[neu.position[0] - 1][index].output *
                      neu.error * self.rate)
                     for index, weight in enumerate(neu.weights)]

        # weight update of the fed-back layer
        for neu in self.hiddens[0]:
            if len(self.stimuli) != len(neu.weights):
                raise RuntimeError("Weight and feedback stimuli dims do not match")
            neu.weights = [weight +
                           self.stimuli[index] * neu.error * self.rate
                           for index, weight in enumerate(neu.weights)]


if __name__ == "__main__":
    bob = RNN(2, 6, 3, 6, 1)
