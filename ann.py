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

import warnings

import numpy as np

# noinspection PyProtectedMember
from .brainforge.layers import _Layer
from .brainforge.optimizers import *


class Network:

    def __init__(self, input_shape=(), name=""):
        # Referencing the data wrapper on which we do the learning
        self.name = name
        # Containers and self-describing variables
        self.layers = []
        self.architecture = []
        self.age = 0
        self.optimizer = None
        self.cost = None
        self.lmbd1 = 0.0
        self.lmbd2 = 0.0
        self.mu = 0.0

        self.N = 0  # X's size goes here
        self.m = 0  # Batch size goes here
        self._finalized = False

        self._add_input_layer(input_shape)

    # ---- Methods for architecture building ----

    def _add_input_layer(self, inshape):
        if not inshape:
            raise RuntimeError("Parameter input_dim must be supplied for the first layer!")
        if isinstance(inshape, int):
            inshape = (inshape,)
        from .brainforge.layers import InputLayer
        self.layers.append(InputLayer(inshape))
        self.layers[-1].connect(to=self, inshape=inshape)
        self.layers[-1].connected = True

    def add(self, layer: _Layer, input_dim=()):
        if len(self.layers) == 0:
            self._add_input_layer(input_dim)
            self.architecture.append(str(self.layers[-1]))

        layer.connect(self, self.layers[-1].outshape)
        self.layers.append(layer)
        self.architecture.append(str(layer))
        layer.connected = True

    def finalize(self, cost, optimizer="sgd", eta=0.1):
        from .util import cost_fns as costs

        for layer in self.layers:
            if layer.trainable:
                layer.optimizer = Adam(layer)
        self.cost = costs[cost] if isinstance(cost, str) else cost
        self._finalized = True

    def pop(self):
        self.layers.pop()
        self.architecture.pop()
        self._finalized = False

    # ---- Methods for model fitting ----

    def fit(self, X, Y, batch_size=20, epochs=10, monitor=(), validation=(), verbose=1, shuffle=True):

        if not self._finalized:
            raise RuntimeError("Architecture not finalized!")

        self.N = X.shape[0]

        for epoch in range(1, epochs+1):
            if shuffle:
                arg = np.arange(X.shape[0])
                np.random.shuffle(arg)
                X, Y = X[arg], Y[arg]
            if verbose:
                print("Epoch {}/{}".format(epoch, epochs))
            self.epoch(X, Y, batch_size, monitor, validation, verbose)

    def fit_csxdata(self, frame, batch_size=20, epochs=10, monitor=(), verbose=1, shuffle=True):
        fanin, outshape = frame.neurons_required
        if fanin != self.layers[0].outshape or outshape != self.layers[-1].outshape:
            errstring = "Network configuration incompatible with supplied dataframe!\n"
            errstring += "fanin: {} <-> InputLayer: {}\n".format(fanin, self.layers[0].outshape)
            errstring += "outshape: {} <-> Net outshape: {}\n".format(outshape, self.layers[-1].outshape)
            raise RuntimeError(errstring)

        X, Y = frame.table("learning")
        validation = frame.table("testing")

        self.fit(X, Y, batch_size, epochs, monitor, validation, verbose, shuffle)

    def epoch(self, X, Y, batch_size, monitor, validation, verbose):

        def print_progress():
            classificaton = "acc" in monitor
            results = self.evaluate(*validation, classify=classificaton)

            chain = "testing cost: {0:.5f}"
            if classificaton:
                tcost, tacc = results
                accchain = "\taccuracy: {0:.2%}".format(tacc)
            else:
                tcost = results[0]
                accchain = ""
            print(chain.format(tcost) + accchain, end="")

        costs = []
        batches = ((X[start:start+batch_size], Y[start:start+batch_size])
                   for start in range(0, self.N, batch_size))

        for bno, (inputs, targets) in enumerate(batches):
            costs.append(self._fit_batch(inputs, targets))
            if verbose:
                done = ((bno * batch_size) + self.m) / self.N
                print("\rDone: {0:>6.1%} Cost: {1: .5f}\t ".format(done, np.mean(costs)), end="")

        if verbose:
            print_progress()
        print()
        self.age += 1
        return costs

    def _fit_batch(self, X, Y, parameter_update=True):
        self.prediction(X)
        self.backpropagation(Y)
        if parameter_update:
            self._parameter_update()

        return self.cost(self.output, Y) / self.m

    def backpropagation(self, Y):
        error = self.cost.derivative(self.layers[-1].output, Y)
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)

    def _parameter_update(self):
        for layer in filter(lambda x: x.trainable, self.layers):
            layer.optimizer(self.m)

    # ---- Methods for forward propagation ----

    def regress(self, X):
        return self.prediction(X)

    def classify(self, X):
        return np.argmax(self.prediction(X), axis=1)

    def prediction(self, X):
        self.m = X.shape[0]
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def evaluate(self, X, Y, classify=True):
        predictions = self.prediction(X)
        cost = self.cost(predictions, Y) / Y.shape[0]
        if classify:
            pred_classes = np.argmax(predictions, axis=1)
            trgt_classes = np.argmax(Y, axis=1)
            eq = np.equal(pred_classes, trgt_classes)
            acc = np.mean(eq)
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
            if layer.trainable:
                layer.shuffle()

    def describe(self, verbose=0):
        if not self.name:
            name = "BrainForge Artificial Neural Network."
        else:
            name = "{}, the Artificial Neural Network.".format(self.name)
        chain = "----------\n"
        chain += name + "\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + "->".join(self.architecture) + "\n"
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
                end = start + layer.nparams
                layer.set_weights(ws[start:end])
                start = end
        else:
            for w, layer in zip(ws, self.layers):
                if not layer.trainable:
                    continue
                layer.set_weights(w)

    def gradient_check(self, X, y, verbose=1, epsilon=1e-5):
        from .util import gradient_check
        if self.age == 0:
            warnings.warn("Performing gradient check on an untrained Neural Network!",
                          RuntimeWarning)
        return gradient_check(self, X, y, verbose=verbose, epsilon=epsilon)

    @property
    def output(self):
        return self.layers[-1].output

    @property
    def weights(self):
        return self.get_weights(unfold=False)

    @weights.setter
    def weights(self, ws):
        self.set_weights(ws, fold=(ws.ndim > 1))

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)
