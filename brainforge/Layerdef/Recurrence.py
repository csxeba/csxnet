import numpy as np

from .Layers import FFLayer
from csxnet.nputils import ravel_to_matrix as ravtm


class RLayer(FFLayer):
    def __init__(self, brain, inputs, neurons, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)
        self.rweights = np.random.randn(neurons, neurons)

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
        pass  # OMGGGGGG


def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)

    # OMG THIS SIMPLY CALCULATES (output - target)...
    # Which by the way is actually the grad of Xent wrt the outweights
    # aka the output error
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.

    # For each output backwards...
    for t in np.arange(T)[::-1]:
        # Calculates the output gradients
        # TODO: I LEFT IT HERE
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
    return [dLdU, dLdV, dLdW]
