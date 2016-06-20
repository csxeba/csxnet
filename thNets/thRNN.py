from ._generic import *
from ._generic import _ThLayerBase


class ThRLayer(_ThLayerBase):
    def __init__(self, neurons, inputs, position, truncation=10):
        _ThLayerBase.__init__(self, inputs, position)

        self._outshape = neurons
        self.input_weights = theano.shared(
            (np.random.randn(self.fanin, neurons) / np.sqrt(self.fanin))
            .astype(floatX), name="R Input Weights"
        )
        self.state_weights = theano.shared(
            (np.random.randn(neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="R State Weights"
        )
        self.biases = theano.shared(
            np.zeros((neurons,), dtype=floatX)
        )

        self.truncate = truncation
        self.params = self.input_weights, self.state_weights, self.biases

    def output(self, inputs, mint):
        U, W, b = self.input_weights, self.state_weights, self.biases

        def step(x_t, y_t_1):
            z = x_t.dot(U) + y_t_1.dot(W) + b
            y_t = T.tanh(z)
            return y_t

        y, updates = theano.scan(step,
                                 sequences=inputs,
                                 truncate_gradient=self.truncate,
                                 outputs_info=[T.zeros_like(inputs)])
        return y

    @property
    def outshape(self):
        return self._outshape


class ThLSTM(_ThLayerBase):
    def __init__(self, neurons, inputs, position, truncation=10):
        _ThLayerBase.__init__(self, inputs, position)
        self._outshape = neurons
        self.input_weights = theano.shared(
            (np.random.randn(4, self.fanin, neurons) / np.sqrt(self.fanin))
            .astype(floatX), name="Input Gates")

        self.state_weights = theano.shared(
            (np.random.randn(4, neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="State Gates")

        self.biases = theano.shared(
            np.zeros((4, neurons), dtype=floatX), name="Biases")

        self.cell_state = theano.shared(
            np.zeros((neurons, neurons), dtype=floatX), name="Cell State")

        self.truncate = truncation
        self.params = self.input_weights, self.state_weights, self.biases

    def output(self, inputs, mint):

        def step(x, prev_o, prev_c):
            preact = prev_o.dot(self.input_weights)
            preact += self.cell_state.dot(self.state_weights) + x

            i = nnet.hard_sigmoid(preact[:, :4])
            f = nnet.hard_sigmoid(preact[:, 4:8])
            o = nnet.hard_sigmoid(preact[:, 8:12])
            c_ = T.tanh(preact[:, 12:])
            state = f * prev_c + i * c_
            output = T.tanh(o * state)

            return output, state

        z = inputs.dot(self.input_weights) + self.biases

        (y, last_c), updates = theano.scan(step,
                                           sequences=z,
                                           truncate_gradient=self.truncate,
                                           outputs_info=(T.zeros((self.outshape,)),
                                                         T.zeros((self.outshape,))))
        return y

    @property
    def outshape(self):
        return self._outshape
