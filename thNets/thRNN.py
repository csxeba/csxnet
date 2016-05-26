import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet

floatX = theano.config.floatX


class ThLSTM:
    def __init__(self, neurons, inputs):
        fanin = np.prod(inputs)
        self.forget_input = theano.shared(
            (np.random.randn(fanin, neurons) / np.sqrt(fanin))
            .astype(floatX), name="Forget Input Gate")
        self.input_input = theano.shared(
            (np.random.randn(fanin, neurons) / np.sqrt(fanin))
            .astype(floatX), name="Input Input Gate")
        self.output_input = theano.shared(
            (np.random.randn(fanin, neurons) / np.sqrt(fanin))
            .astype(floatX), name="Output Input Gate")

        self.forget_state = theano.shared(
            (np.random.randn(neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="Forget State Gate")
        self.input_state = theano.shared(
            (np.random.randn(neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="Input State Gate")
        self.output_state = theano.shared(
            (np.random.randn(neurons, neurons) / np.sqrt(neurons))
            .astype(floatX), name="Output State Gate")

        self.forget_bias = theano.shared(
            np.zeros((neurons,), dtype=floatX),
            name="Forget Bias")
        self.input_bias = theano.shared(
            np.zeros((neurons,), dtype=floatX),
            name="Input Bias")
        self.output_bias = theano.shared(
            np.zeros((neurons,), dtype=floatX),
            name="Output Bias")

        self.cell_state = theano.shared(
            np.zeros(neurons, neurons).astype(floatX),
            name="Cell State")

    def output(self, inputs, mint):
        i = nnet.sigmoid(
            inputs.dot(self.input_input) +

        )