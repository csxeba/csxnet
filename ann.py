"""
Neural Network Framework on top of NumPy
Copyright (C) 2016  Csaba Gór

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

import abc

import numpy as np

from .util import act_fns, cost_fns

from csxdata.utilities.pure import niceround


class NeuralNetworkBase(abc.ABC):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, name: str):
        # Referencing the data wrapper on which we do the learning
        self.data = data
        self.fanin, self.outsize = data.neurons_required

        self.name = name
        self.N = data.N

        # Parameters required for SGD
        self.eta = eta
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.mu = mu

        # Containers and self-describing variables
        self.layers = []
        self.architecture = []
        self.age = 0
        self.name = ""

    @abc.abstractmethod
    def fit(self, batch_size, epochs): raise NotImplementedError

    @abc.abstractmethod
    def _epoch(self, batch_size: int): raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, on: str): raise NotImplementedError

    @abc.abstractmethod
    def predict(self, questions: np.ndarray): raise NotImplementedError

    @abc.abstractmethod
    def describe(self): raise NotImplementedError


class Network(NeuralNetworkBase):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, cost, name=""):
        from .brainforge.layers import InputLayer

        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu, name)

        self.m = 0  # Batch size goes here

        if isinstance(cost, str):
            self.cost = cost_fns[cost]
        else:
            self.cost = cost

        self.layers.append(InputLayer(brain=self, inshape=self.fanin))
        self.architecture.append("In: {}".format(self.fanin))
        self.predictor = None
        self.encoder = None

        self.finalized = False

    # ---- Methods for architecture building ----

    def add_conv(self, fshape=(3, 3), n_filters=1, stride=1, activation=act_fns.tanh):
        from .brainforge.layers import Experimental
        fshape = [self.fanin[0]] + list(fshape)
        args = (self, fshape, self.layers[-1].outshape, n_filters, stride, len(self.layers), activation)
        self.layers.append(Experimental.ConvLayer(*args))
        self.architecture.append("{}x{}x{} Conv: {}".format(fshape[0], fshape[1], n_filters, str(activation)[:4]))
        # brain, fshape, fanin, num_filters, stride, position, activation="sigmoid"

    def add_pool(self, pool=2):
        from .brainforge.layers import Experimental
        args = (self, self.layers[-1].outshape, (pool, pool), pool, len(self.layers))
        self.layers.append(Experimental.PoolLayer(*args))
        self.architecture.append("{} Pool".format(pool))
        # brain, fanin, fshape, stride, position

    def add_fc(self, neurons, activation="tanh"):
        from .brainforge.layers import DenseLayer
        inpts = np.prod(self.layers[-1].outshape)
        args = (self, inpts, neurons, len(self.layers), activation)
        self.layers.append(DenseLayer(*args))
        self.architecture.append("{} Dense: {}".format(neurons, str(activation)[:4]))
        # brain, inputs, neurons, position, activation

    def add_drop(self, neurons, dropchance=0.25, activation="tanh"):
        from .brainforge.layers import DropOut
        args = (self, np.prod(self.layers[-1].outshape), neurons, dropchance, len(self.layers), activation)
        self.layers.append(DropOut(*args))
        self.architecture.append("{} Drop({}): {}".format(neurons, round(dropchance, 2), str(activation)[:4]))
        # brain, inputs, neurons, dropout, position, activation

    def add_rec(self, neurons, activation="tanh"):
        from .brainforge.layers import RLayer
        inpts = np.prod(self.layers[-1].outshape)
        args = self, inpts, neurons, len(self.layers), activation
        self.layers.append(RLayer(*args))
        self.architecture.append("{} RecL: {}".format(neurons, activation[:4]))
        # brain, inputs, neurons, time_truncate, position, activation

    def finalize_architecture(self, activation="sigmoid"):
        from .brainforge.layers import DenseLayer
        pargs = (self, np.prod(self.layers[-1].outshape), self.outsize, len(self.layers), activation)
        self.predictor = DenseLayer(*pargs)
        self.layers.append(self.predictor)
        self.architecture.append("{} Dense: {}".format(self.outsize, str(activation)[:4]))
        self.finalized = True

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self.finalized = False

    # ---- Methods for model fitting ----

    def fit(self, batch_size=20, epochs=10, verbose=1, monitor=()):

        if not self.finalized:
            raise RuntimeError("Architecture not finalized!")

        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs))
            self._epoch(batch_size, verbose, monitor)

    def _epoch(self, batch_size=20, verbose=1, monitor=()):

        def print_progress():
            tcost, tacc = self.evaluate("testing")
            tcost = niceround(tcost, 5)
            tacc = niceround(tacc * 100, 4)
            print("testing cost: {};\taccuracy: {}%".format(tcost, tacc), end="")

        costs = []
        for bno, (inputs, targets) in enumerate(self.data.batchgen(batch_size)):
            costs.append(self._fit_batch(inputs, targets))
            if verbose:
                done_percent = int(100 * (((bno + 1) * batch_size) / self.N))
                print("\r{}%:\tCost: {}\t ".format(done_percent, niceround(np.mean(costs), 5)), end="")
        if "acc" in monitor:
            print_progress()
        print()
        return costs

    def _fit_batch(self, X, y):
        """
        This method coordinates the fitting of parameters (learning and encoding).

        Backprop and weight update is implemented layer-wise and
        could be somewhat parallelized.
        Backpropagation is done a bit unorthodoxically. Each
        layer computes the error of the previous layer and the
        backprop methods return with the computed error array

        :param X: NumPy array of stimuli
        :param y: NumPy array of (embedded) targets
        :return: cost calculated on the current batch
        """

        self.m = X.shape[0]

        # Forward pass
        for layer in self.layers:
            X = layer.feedforward(X)
        # Calculate the cost derivative
        self.layers[-1].receive_error(self.cost.derivative(self.layers[-1].output, y))
        # Backpropagate the errors
        for layer, prev_layer in zip(self.layers[-1:0:-1], self.layers[-2::-1]):
            prev_layer.receive_error(layer.backpropagation())
            layer.weight_update()
        return self.cost(self.layers[-1].output, y)

    # ---- Methods for forward propagation ----

    def predict(self, X):
        if self.data.type == "classification":
            return self.predict_class(X)
        elif self.data.type == "regression":
            return self.predict_raw(X)
        else:
            raise TypeError("Unsupported Dataframe Type")

    def predict_class(self, X):
        """
        Coordinates prediction (feedforwarding outside the learning phase)

        :param X: numpy.ndarray representing a batch of inputs
        :return: numpy.ndarray: 1D array of predictions
        """
        return np.argmax(self.predict_raw(X), axis=1)

    def predict_raw(self, X):
        for layer in self.layers:
            X = layer.predict(X)
        return X

    def evaluate(self, on="testing", accuracy=True):
        """
        Calculates the network's prediction accuracy

        :param on: cross-validation is implemented, dataset can be chosen here
        :param accuracy: if True, the class prediction accuracy is calculated.
        :return: rate of right answers
        """

        X, y = self.data.table(on, shuff=True, m=self.data.n_testing)
        predictions = self.predict_raw(X)
        cost = self.cost(predictions, y)
        if accuracy:
            pred_classes = np.argmax(predictions, axis=1)
            trgt_classes = np.argmax(y, axis=1)
            eq = np.equal(pred_classes, trgt_classes)
            acc = np.average(eq)
            return cost, acc
        return cost

    # ---- Some utilities ----

    def save(self, path):
        import pickle

        fl = open(path, mode="wb")
        print("Saving brain object to", path)
        pickle.dump(self, fl)
        fl.close()

    def shuffle(self):
        for layer in self.layers:
            layer.shuffle()

    def describe(self, verbose=0):
        if not self.name:
            name = "CsxNet BrainForge Artificial Neural Network."
        else:
            name = "{}, the Artificial Neural Network.".format(self.name)
        chain = "----------\n"
        chain += name + "\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + str(self.architecture) + "\n"
        chain += "----------"
        if verbose:
            print(chain)
        else:
            return chain

    def dream(self, matrix):
        """Reverse-feedforward"""
        assert not all(["C" in l for l in self.architecture]), "Convolutional dreaming not <yet> supported!"
        assert all([(isinstance(layer.activation, act_fns.sigmoid)) or (layer.activation is act_fns.sigmoid)
                    for layer in self.layers[1:]]), "Only Sigmoid is supported!"

        from csxdata.utilities.nputils import logit

        print("Warning! Network.dream() is highly experimental and possibly buggy!")

        for layer in self.layers[-1:0:-1]:
            matrix = logit(matrix.dot(layer.weights.T))
        return matrix


class FeedForwardNet(Network):
    """
    Layerwise representation of a Feed Forward Neural Network

    Learning rate is given by the keyword argument <rate>.
    The neural network architecture is given by <hiddens>.
    Multiple hidden layers may be defined. Input and output neurons
    are calculated from the shape of <data>
    """

    def __init__(self, hiddens, data, eta, lmbd1=0.0, lmbd2=0.0, mu=0.0,
                 cost=cost_fns.xent, activation=act_fns.tanh):
        Network.__init__(self, data=data, eta=eta, lmbd1=lmbd1, lmbd2=lmbd2, mu=mu, cost=cost)

        if isinstance(hiddens, int):
            hiddens = (hiddens,)

        self.layout = tuple(list(self.layers[0].outshape) + list(hiddens) + [self.outsize])

        for neu in hiddens:
            self.add_fc(neurons=neu, activation=activation)
        self.finalize_architecture(activation=act_fns.sigmoid)
