import numpy as np

from .Layers import FFLayer
from csxnet.nputils import ravel_to_matrix as ravtm


class RLayer(FFLayer):
    def __init__(self, brain, inputs, neurons, time_truncate, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)
        self.rweights = np.random.randn(neurons, neurons)
        self.time_truncate = time_truncate
        self.d_weights = np.zeros_like(self.weights)
        self.d_rweights = np.zeros_like(self.rweights)

    def feedforward(self, questions):
        self.inputs = ravtm(questions)
        time = questions.shape[0]
        self._output = np.zeros((time+1, self.outshape))
        for t in range(time):
            self.output[t] = self.activation(
                np.dot(self.inputs[t], self.weights[t]) +
                np.dot(self.output[t-1], self.rweights)
            )
        return self.output

    def backpropagation(self):
        """Backpropagation through time (BPTT)"""
        T = self.error.shape[0]
        self.d_weights = np.zeros(self.weights.shape)
        self.d_rweights = np.zeros(self.rweights.shape)
        prev_error = np.zeros_like(self.inputs)
        for t in range(0, T, step=-1):
            t_delta = self.error[t]
            for bptt in range(max(0, t-self.time_truncate), t+1, step=-1):
                # TODO: check the order of parameters. Transposition possibly needed somewhere
                self.d_rweights += np.outer(t_delta, self.output[bptt-1])
                self.d_weights += np.dot(self.d_weights, self.inputs) + t_delta
                t_delta = self.rweights.dot(t_delta) * self.activation.derivative(self.output[bptt-1])
            prev_error[t] = t_delta

    def receive_error(self, error):
        """
        Transforms the received error tensor to adequate shape and stores it.

        :param error: T x N shaped, where T is time and N is the number of neurons
        :return: None
        """
        self.error = ravtm(error)

    def weight_update(self):
        self.weights += self.brain.eta * self.d_weights
        self.rweights += self.brain.eta * self.d_rweights


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
    # Which by the way is the grad of Xent wrt to the outweights
    # aka the output error
    # *provided that target is not converted to 1-hot vectors
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.

    for t in np.arange(T)[::-1]:
        # This is the sum of the output weights' gradients
        dLdV += np.outer(delta_o[t], s[t].T)
        # backpropagating to the hiddens (Weights * Er) * tanh'(Z)
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time (for at most self.time_truncate steps)
        for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step - 1])
            dLdU[:, x[bptt_step]] += delta_t
            # Update delta for next step
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
    return [dLdU, dLdV, dLdW]


def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, self.word_dim))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        # aka self.U.dot(x)
        s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
        o[t] = softmax(self.V.dot(s[t]))
    return [o, s]
