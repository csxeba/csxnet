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
import warnings

import numpy as np

from .util import activations, costs

from csxdata.utilities.pure import niceround


class NeuralNetworkBase(abc.ABC):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, name: str, fanin=0, outsize=0, N=0):

        # Referencing the data wrapper on which we do the learning
        self.data = data
        self.name = name
        if data is not None:
            self.N = data.N
            self.fanin, self.outsize = data.neurons_required
        else:
            self.fanin, self.outsize, self.N = fanin, outsize, N

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
    def fit(self, batch_size: int): pass

    @abc.abstractmethod
    def evaluate(self, on: str): pass

    @abc.abstractmethod
    def predict(self, questions: np.ndarray): pass

    @abc.abstractmethod
    def describe(self): pass


class Network(NeuralNetworkBase):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, cost, name=""):
        from .brainforge.layers import InputLayer

        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu, name)

        self.m = 0  # Batch size goes here

        if isinstance(cost, str):
            self.cost = costs[cost]
        else:
            self.cost = cost

        self.layers.append(InputLayer(brain=self, inshape=self.fanin))
        self.architecture.append("In: {}".format(self.fanin))
        self.predictor = None
        self.encoder = None

        self.finalized = False

    # ---- Methods for architecture building ----

    def add_conv(self, fshape=(3, 3), n_filters=1, stride=1, activation=activations.tanh):
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

    def add_fc(self, neurons, activation=activations.tanh):
        from .brainforge.layers import DenseLayer
        inpts = np.prod(self.layers[-1].outshape)
        args = (self, inpts, neurons, len(self.layers), activation)
        self.layers.append(DenseLayer(*args))
        self.architecture.append("{} Dense: {}".format(neurons, str(activation)[:4]))
        # brain, inputs, neurons, position, activation

    def add_drop(self, neurons, dropchance=0.25, activation=activations.tanh):
        from .brainforge.layers import DropOut
        args = (self, np.prod(self.layers[-1].outshape), neurons, dropchance, len(self.layers), activation)
        self.layers.append(DropOut(*args))
        self.architecture.append("{} Drop({}): {}".format(neurons, round(dropchance, 2), str(activation)[:4]))
        # brain, inputs, neurons, dropout, position, activation

    def add_rec(self, neurons, time_truncate=5, activation=activations.tanh):
        from .brainforge.layers import Experimental
        inpts = np.prod(self.layers[-1].outshape)
        args = self, inpts, neurons, time_truncate, len(self.layers), activation
        self.layers.append(Experimental.RLayer(*args))
        self.architecture.append("{} RecL(time={}): {}".format(neurons, time_truncate, activation[:4]))
        # brain, inputs, neurons, time_truncate, position, activation

    def finalize_architecture(self, activation=activations.sigmoid):
        from .brainforge.layers import DenseLayer
        pargs = (self, np.prod(self.layers[-1].outshape), self.outsize, len(self.layers), activation)
        self.predictor = DenseLayer(*pargs)
        self.layers.append(self.predictor)
        self.architecture.append("{} Dense: {}".format(self.outsize, str(activation)[:4]))
        self.finalized = True

    # ---- Methods for model fitting ----

    def fit(self, batch_size=20, epochs=10, verbose=1, monitor=()):

        def do_batch(lessons):
            self.m = lessons[0].shape[0]
            cst = self._fit(lessons)
            return cst

        def sanity_check():
            if not self.finalized:
                raise RuntimeError("Architecture not finalized!")
            if self.layers[-1] is not self.predictor:
                self._insert_predictor()

        def print_progress():
            tcost, tacc = self.evaluate("testing")
            tcost = niceround(tcost, 5)
            tacc = niceround(tacc * 100, 4)
            print("testing cost: {};\taccuracy: {}%".format(tcost, tacc), end="")

        sanity_check()
        cost = []
        for epoch in range(1, epochs+1):
            cost += [do_batch(bno, batch) for bno, batch in enumerate(self.data.batchgen(batch_size))]
            for i, batch in enumerate(self.data.batchgen(batch_size)):
                cost.append(do_batch(batch))
                done_percent = int(100 * (((batch_no + 1) * batch_size) / self.data.N))
                if verbose:
                    print("\rEpoch: {}")

            if "acc" in monitor:
                print_progress()
            print()

    def learn(self, batch_size=20, verbose=1, monitor=("acc",)):

        def do_batch(batch_no, lessons):
            self.m = lessons[0].shape[0]
            cst = self._fit(lessons)
            if verbose:
                done_percent = int(100 * (((batch_no + 1) * batch_size) / self.data.N))
                print("\r{}%:\tCost: {}\t "
                      .format(done_percent, niceround(cst, 5)), end="")

            return cst

        def sanity_check():
            if not self.finalized:
                raise RuntimeError("Architecture not finalized!")
            if self.layers[-1] is not self.predictor:
                self._insert_predictor()

        def print_progress():
            tcost, tacc = self.evaluate("testing")
            tcost = niceround(tcost, 5)
            tacc = niceround(tacc * 100, 4)
            print("testing cost: {};\taccuracy: {}%".format(tcost, tacc), end="")

        warnings.warn("learn() is deprecated! Please switch to fit()!", DeprecationWarning)

        sanity_check()
        for bno, batch in enumerate(self.data.batchgen(batch_size)):
            do_batch(bno, batch)
        if "acc" in monitor:
            print_progress()
        print()

    def _fit(self, learning_table):
        """
        This method coordinates the fitting of parameters (learning and encoding).

        Backprop and weight update is implemented layer-wise and
        could be somewhat parallelized.
        Backpropagation is done a bit unorthodoxically. Each
        layer computes the error of the previous layer and the
        backprop methods return with the computed error array

        :param learning_table: tuple of 2 numpy arrays: (stimuli, _embedments)
        :return: None
        """

        questions, targets = learning_table
        # Forward pass
        for layer in self.layers:
            questions = layer.feedforward(questions)
        # Calculate the cost derivative
        last = self.layers[-1]
        last.receive_error(self.cost.derivative(last.output, targets))
        # Backpropagate the errors
        for layer, prev_layer in zip(self.layers[-1:0:-1], self.layers[-2::-1]):
            prev_layer.receive_error(layer.backpropagation())
        # Update weights
        for layer in self.layers[1:]:
            layer.weight_update()
        return self.cost(last.output, targets)

    # ---- Methods for forward propagation ----

    def predict(self, questions):
        if self.data.type == "classification":
            return self.predict_class(questions)
        elif self.data.type == "regression":
            return self.predict_raw(questions)
        else:
            raise TypeError("Unsupported Dataframe Type")

    def predict_class(self, questions):
        """
        Coordinates prediction (feedforwarding outside the learning phase)

        The layerwise implementations of <_evaluate> don't have any side-effects
        so prediction is a candidate for parallelization.

        :param questions: numpy.ndarray representing a batch of inputs
        :return: numpy.ndarray: 1D array of predictions
        """
        return np.argmax(self.predict_raw(questions), axis=1)

    def predict_raw(self, questions):
        for layer in self.layers:
            questions = layer.predict(questions)
        return questions

    def evaluate(self, on="testing", accuracy=True):
        """
        Calculates the network's prediction accuracy

        :param on: cross-validation is implemented, dataset can be chosen here
        :param accuracy: if True, the class prediction accuracy is calculated.
        :return: rate of right answers
        """

        questions, targets = self.data.table(on, shuff=True, m=self.data.n_testing)
        predictions = self.predict_raw(questions)
        cost = self.cost(predictions, targets)
        if accuracy:
            pred_classes = np.argmax(predictions, axis=1)
            trgt_classes = np.argmax(targets, axis=1)
            eq = np.equal(pred_classes, trgt_classes)
            acc = np.average(eq)
            return cost, acc
        return cost

    # ---- Private helper methods ----

    def _insert_encoder(self):
        if not isinstance(self.cost, costs["mse"]) or self.cost is not costs["mse"]:
            print("Cost function not supported in autoencoding! Falling back to MSE!")
            self.cost = costs["mse"]
        self.layers[-1] = self.encoder
        print("Inserted encoder as output layer!")

    def _insert_predictor(self):
        self.layers[-1] = self.predictor
        print("Inserted predictor as output layer!")

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
        assert all([(isinstance(layer.activation, activations.sigmoid)) or (layer.activation is activations.sigmoid)
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
                 cost=costs.xent, activation=activations.tanh):
        Network.__init__(self, data=data, eta=eta, lmbd1=lmbd1, lmbd2=lmbd2, mu=mu, cost=cost)

        if isinstance(hiddens, int):
            hiddens = (hiddens,)

        self.layout = tuple(list(self.layers[0].outshape) + list(hiddens) + [self.outsize])

        for neu in hiddens:
            self.add_fc(neurons=neu, activation=activation)
        self.finalize_architecture(activation=activations.sigmoid)
