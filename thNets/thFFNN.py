import abc

import numpy as np
from theano.tensor.signal.downsample import max_pool_2d

from ._generic import *

floatX = theano.config.floatX

print("floatX is set to <{}>".format(floatX))
print("Device used: <{}>".format(theano.config.device))


class _ThLayerBase(abc.ABC):
    def __init__(self, inshape, position):
        self.fanin = np.prod(inshape)
        self.inshape = inshape
        self.position = position

    @abc.abstractmethod
    def output(self, intputs, mint): pass

    @property
    def outshape(self):
        return None


class ThConvPoolLayer(_ThLayerBase):
    def __init__(self, conv, filters, pool, inshape, position):
        _ThLayerBase.__init__(self, inshape, position)
        channel, ix, iy = inshape

        assert ix == iy, "Only square convolution is supported!"
        assert ((ix - conv) + 1) % pool == 0, "Non-integer ConvPool output shape!"

        osh = ((ix - conv) + 1) // pool
        self._outshape = osh, osh, filters
        self.fshape = filters, channel, conv, conv

        self.weights = theano.shared(
            (np.random.randn(*self.fshape) /
             np.sqrt(self.fanin)).astype(floatX),
            name="{}. ConvFilters".format(position)
        )

        self.biases = theano.shared(np.zeros((filters,), dtype=floatX))

        self.params = [self.weights, self.biases]

        self.pool = pool

    def output(self, inputs, mint):
        cact = nnet.conv2d(inputs, self.weights)
        pact = max_pool_2d(cact, ds=(self.pool, self.pool), ignore_border=True)
        return T.tanh(pact + self.biases.dimshuffle("x", 0, "x", "x"))

    @property
    def outshape(self):
        return self._outshape


class ThFCLayer(_ThLayerBase):
    def __init__(self, neurons, inshape, position, activation="sigmoid"):
        _ThLayerBase.__init__(self, inshape, position)
        self._outshape = neurons
        self.activation = {"sigmoid": nnet.sigmoid, "tanh": T.tanh, "softmax": nnet.softmax}[activation.lower()]
        self.weights = theano.shared((np.random.randn(self.fanin, neurons) / np.sqrt(self.fanin)).astype(floatX),
                                     name="{}. FCweights".format(position))
        self.biases = theano.shared((np.zeros((neurons,), dtype=floatX)),
                                    name="{}. FCbiases".format(self.position))
        self.params = [self.weights, self.biases]

    def output(self, inputs, mint):
        i = T.reshape(inputs, (mint, self.fanin))
        return self.activation(i.dot(self.weights) + self.biases)

    @property
    def outshape(self):
        return self._outshape


class ThOutputLayer(ThFCLayer):
    def __init__(self, neurons, inshape, position):
        ThFCLayer.__init__(self, neurons, inshape, position, activation="softmax")


class ThDropoutLayer(ThFCLayer):
    def __init__(self, neurons, inshape, dropchance, position, activation="sigmoid"):
        del dropchance
        ThFCLayer.__init__(neurons, inshape, position, activation)
        print("Dropout not implemented yet, falling back to ThFCLayer!")
