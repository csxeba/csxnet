import numpy as np

from .Layers import FFLayer
from csxnet.nputils import ravel_to_matrix as ravtm


class RLayer(FFLayer):
    def __init__(self, brain, inputs, neurons, bptt_truncate, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)
        self.rweights = np.random.randn(neurons, neurons)
        self.bptt_truncate = bptt_truncate

    def feedforward(self, questions):
        self.inputs = ravtm(questions)
        time = questions.shape[0]
        self.output = np.zeros((time+1, self.outshape))
        for t in range(time):
            self.output[t] = self.activation(
                np.dot(self.inputs[t], self.weights[t]) +
                np.dot(self.output[t-1], self.rweights)
            )
        return self.output

    def backpropagation(self):
        """Backpropagation through time (BPTT)"""
        raise NotImplementedError(":(")

    def receive_error(self, error):
        self.error = ravtm(error)

    def weight_update(self):
        dLdU = np.zeros(self.weights.shape)
        dLdW = np.zeros(self.rweights.shape)

        for bptt in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
            np.outer(self.error, self.output[bptt - 1])


def bptt_reference(self, x, y):
    T = len(y)
    # Perform forward propagation
    # Catch network output and hidden output
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)

    # OMG THIS SIMPLY CALCULATES (output - target*)...
    # Which by the way is the grad of Xent wrt the outweights
    # aka the output error
    # * provided that target is an array of 1-hot vectors
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.

    for t in np.arange(T)[::-1]:
        # This is the sum of the weights' gradients
        dLdV += np.outer(delta_o[t], s[t].T)
        # backpropagating to the hiddens (W * Er) * tanh'(Z)
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
    return [dLdU, dLdV, dLdW]
