import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet


floatX = theano.config.floatX


class ThLSTM:
    def __init__(self, neurons, inputs):
        fanin = np.prod(inputs)
        self.outshape = neurons
        self.input_weights = theano.shared(
            (np.random.randn(4, fanin, neurons) / np.sqrt(fanin))
            .astype(floatX), name="Input Gates")

        self.state_weights = theano.shared(
            (np.random.randn(4, neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="State Gates")

        self.biases = theano.shared(
            np.zeros((4, neurons), dtype=floatX), name="Biases")

        self.cell_state = theano.shared(
            np.zeros(neurons, neurons).astype(floatX), name="Cell State")

    def output(self, inputs, mint):

        def step(prev_o, prev_c):
            preact = prev_o.dot(self.input_weights)
            preact += self.cell_state.dot(self.state_weights) + self.biases

            i = nnet.hard_sigmoid(preact[..., :4])
            f = nnet.hard_sigmoid(preact[..., 4:8])
            o = nnet.hard_sigmoid(preact[..., 8:12])
            c_ = T.tanh(preact[..., 12:])
            state = f * prev_c + i * c_
            output = T.tanh(o * state)

            return output, state

        (last_o, last_c), updates = theano.scan(step,
                                                sequences=inputs,
                                                outputs_info=(T.zeros((mint, self.outshape)),
                                                            T.zeros((mint, self.outshape))))
        return last_o
