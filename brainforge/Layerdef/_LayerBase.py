"""
Abstract base classes for the layer implementations.
Why did I create these? No idea...
"""

import numpy as np

from ..Utility.utility import outshape, calcsteps


class _LayerBase(object):
    """Abstract base class for all layer type classes"""
    def __init__(self, brain, position, activation):
        self.brain = brain
        self.position = position
        self.activation = activation()

    def feedforward(self, stimuli): pass

    def predict(self, stimuli): pass

    def backpropagation(self): pass

    def weight_update(self): pass

    def receive_error(self, error_vector): pass


class _VecLayer(_LayerBase):
    """Base class for layer types, which operate on tensors
     and are sparsely connected"""
    def __init__(self, brain, inshape: tuple, fshape: tuple,
                 stride: int, position: int,
                 activation: type):
        _LayerBase.__init__(self, brain, position, activation)

        if len(fshape) != 3:
            fshape = ("NaN", fshape[0], fshape[1])

        self.fshape, self.stride = fshape, stride

        self.inshape = inshape
        self.outshape = outshape(inshape, fshape, stride)

        self.inputs = np.zeros(inshape)
        self.excitation = np.zeros(self.outshape)
        self.output = np.zeros(self.outshape)
        self.error = np.zeros(self.outshape)

        self.coords = calcsteps(inshape, fshape, stride)


class _FCLayer(_LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, brain, neurons: int, position: int, activation: type):
        _LayerBase.__init__(self, brain, position, activation)

        self.neurons = neurons
        self.outshape = (neurons,)
        self.inputs = None

        self.excitation = np.zeros((neurons,))
        self.output = np.zeros((neurons,))
        self.error = np.zeros((neurons,))
