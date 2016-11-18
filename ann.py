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


class NeuralNetworkBase(abc.ABC):

    def __init__(self, name: str):
        # Referencing the data wrapper on which we do the learning
        self.name = name
        # Containers and self-describing variables
        self.layers = []
        self.architecture = []
        self.age = 0
        self.m = 0
        self.cost = None
        self.lmbd1 = 0.0
        self.lmbd2 = 0.0
        self.mu = 0.0

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

    def __init__(self, input_shape, name=""):
        from .brainforge.layers import InputLayer
        NeuralNetworkBase.__init__(self, name)

        self.layers.append(InputLayer(self, shape=input_shape))
        self.m = 0  # Batch size goes here
        self._finalized = False

    # ---- Methods for architecture building ----

    def add(self, layer):
        self.layers.append(layer)
        self.architecture.append(str(layer))

    def add_conv(self, fshape=(3, 3), n_filters=1, stride=1, activation="tanh"):
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

    def _add_recurrent(self, neurons, activation="tanh", return_seq=False, echo=False, p=0.0, lstm=False):
        inpts = self.layers[-1].outshape[-1]
        args = [self, inpts, neurons, len(self.layers), activation, return_seq]
        if lstm:
            from .brainforge.layers import LSTM
            self.layers.append(LSTM(*args))
            self.architecture.append("{} LSTM: {}".format(neurons, activation[:4]))
            return
        if not echo:
            from .brainforge.layers import RLayer
            self.layers.append(RLayer(*args))
            self.architecture.append("{} RecL: {}".format(neurons, activation[:4]))
            # brain, inputs, neurons, time_truncate, position, activation
        else:
            from .brainforge.layers import EchoLayer
            self.layers.append(EchoLayer(*(args + [p])))
            self.architecture.append("{} Echo({}): {}".format(neurons, p, activation[:4]))
            # brain, inputs, neurons, time_truncate, position, activation

    def add_reclayer(self, neurons, activation="tanh", return_seq=False):
        self._add_recurrent(neurons, activation, return_seq)

    def add_lstm(self, neurons, activation="tanh", return_seq=False):
        self._add_recurrent(neurons, activation, return_seq, lstm=True)

    def add_echo(self, neurons, activation="tanh", return_seq=False, p=0.1):
        self._add_recurrent(neurons, activation, return_seq, echo=True, p=p)

    def finalize(self, cost, optimizer="sgd", lambda1=0.0, lambda2=0.0, mu=0.0):
        from .util import cost_fns as costs

        if optimizer != "sgd":
            raise RuntimeError("Only SGD is supported at the moment!")

        for layer in self.layers:
            layer.connect(self)

        self.cost = costs[cost] if isinstance(cost, str) else cost
        self.lmbd1 = lambda1
        self.lmbd2 = lambda2
        self.mu = mu
        self._finalized = True

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    # ---- Methods for model fitting ----

    def fit(self, batch_size=20, epochs=10, verbose=1, monitor=()):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        for epoch in range(1, epochs+1):
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs))
            self._epoch(batch_size, verbose, monitor)

    def _epoch(self, batch_size=20, verbose=1, monitor=()):

        def print_progress():
            tcost, tacc = self.evaluate("testing")
            print("testing cost: {0:.5f};\taccuracy: {1:.2%}".format(tcost, tacc), end="")

        costs = []
        for bno, (inputs, targets) in enumerate(self.data.batchgen(batch_size), start=1):
            costs.append(self._fit_batch(inputs, targets))
            if verbose:
                done = (bno * batch_size) / self.N
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(done, np.mean(costs)), end="")
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

        self._forward_pass(X)
        self._backward_pass(y)
        self._parameter_update()

        endcost = self.cost(self.output, y) / self.m
        return endcost

    def _forward_pass(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)

    def _backward_pass(self, y):
        self.layers[-1].receive_error(self.cost.derivative(self.layers[-1].output, y))
        for layer, prev_layer in zip(self.layers[-1:0:-1], self.layers[-2::-1]):
            prev_layer.receive_error(layer.backpropagation())

    def _parameter_update(self):
        for layer in self.layers[-1:0:-1]:
            layer.weight_update()

    # ---- Methods for forward propagation ----

    def predict(self, X):
        if self.data.type == "classification":
            return self.predict_class(X)
        elif self.data.type == "regression":
            return self.predict_raw(X)
        elif self.data.type == "sequence":
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
        """Make predictions given an input matrix"""
        for layer in self.layers:
            X = layer.predict(X)
        return X

    def evaluate(self, on="testing", accuracy=True):
        """
        Calculates the network's prediction accuracy

        :param on: cross-validation is implemented, dataset can be chosen here
        :param accuracy: if True, the class prediction accuracy is calculated.
        :return: cost and rate of right answers
        """

        X, y = self.data.table(on, shuff=True, m=self.data.n_testing)
        predictions = self.predict_raw(X)
        cost = self.cost(predictions, y) / y.shape[0]
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

    def get_weights(self, unfold=True):
        ws = [layer.get_weights(unfold=unfold) for layer in self.layers if layer.trainable]
        return np.concatenate(ws) if unfold else ws

    def set_weights(self, ws, fold=True):
        if fold:
            start = 0
            for layer in self.layers:
                if not layer.trainable:
                    continue
                end = start + np.prod(layer.weights.shape)
                layer.set_weights(ws[start:end])
                start += end
        else:
            for w, layer in zip(ws, self.layers):
                if not layer.trainable:
                    continue
                layer.set_weights(w)

    def gradient_check(self, X=None, y=None, verbose=1):

        def get_data():
            nX, ny = self.data.table("testing", m=20, shuff=False)
            if nX is None and ny is None:
                nX, ny = self.data.table("learning", m=20, shuff=False)
            if X is None and y is not None:
                return nX
            elif y is None and X is not None:
                return ny
            elif X is None and y is None:
                return nX, ny

        if X is None or y is None:
            X, y = get_data()

        if self.age == 0:
            print("Performing gradient check on an untrained Neural Network!")
            print("This can lead to numerical unstability. Training 1 epoch now!")
            self._epoch(20, verbose=1)

        from .util import gradient_check

        return gradient_check(self, X, y, verbose=verbose)

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def weights(self):
        return self.get_weights(unfold=False)

    @weights.setter
    def weights(self, ws):
        self.set_weights(ws, fold=(ws.ndim > 1))


class FeedForwardNet(Network):
    """
    Layerwise representation of a Feed Forward Neural Network

    Learning rate is given by the keyword argument <rate>.
    The neural network architecture is given by <hiddens>.
    Multiple hidden layers may be defined. Input and output neurons
    are calculated from the shape of <data>
    """

    def __init__(self, neurons, data, eta, lmbd1=0.0, lmbd2=0.0, mu=0.0,
                 cost="xent", activation="tanh", output_activation="sigmoid"):
        Network.__init__(self, data=data, eta=eta, lmbd1=lmbd1, lmbd2=lmbd2, mu=mu, cost=cost)

        if isinstance(neurons, int):
            neurons = (neurons,)

        self.layout = tuple([self.layers[0].outshape] + list(neurons) + [self.outsize])

        for neu in neurons:
            self.add_fc(neurons=neu, activation=act_fns[activation])
        self.layers[-1].activation = act_fns[output_activation]
        self.finalize(cost=cost, lambda1=lmbd1, lambda2=lmbd2, mu=mu)

    @property
    def weights(self):
        weights = np.array([])
        for layer in self.layers[1:]:
            weights = np.concatenate((weights, layer.weights.ravel()))
        return weights

    @weights.setter
    def weights(self, ws):
        start = 0
        for layer in self.layers[1:]:
            end = start + np.prod(layer.weights.shape)
            layer.weights = ws[start:end].reshape(layer.weights.shape)
            start += end
