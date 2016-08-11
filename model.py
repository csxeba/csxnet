"""
Neural Network Framework on top of NumPy and/or Theano.
Copyright (C) 2016 Csaba Gór

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

from .brainforge.activations import *
from .brainforge.layers import *
from .brainforge.cost import *
from .thNets.thFFNN import *
from .thNets.thRNN import *


class NeuralNetworkBase(abc.ABC):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float):

        # Referencing the data wrapper on which we do the learning
        self.data = data
        self.N = data.N
        self.fanin, self.outsize = data.neurons_required()

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
    def learn(self, batch_size: int): pass

    @abc.abstractmethod
    def evaluate(self, on: str): pass

    @abc.abstractmethod
    def predict(self, questions: np.ndarray): pass

    @abc.abstractmethod
    def describe(self): pass


class Network(NeuralNetworkBase):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, cost):

        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu)

        self.error = float()
        self.m = int()  # Batch size goes here

        if isinstance(cost, str):
            self.cost = fromstring[cost]
        else:
            self.cost = cost

        self.layers.append(InputLayer(brain=self, inshape=self.fanin))
        self.architecture.append("In: {}".format(self.fanin))
        self.predictor = None
        self.encoder = None

        self.finalized = False

    # ---- Methods for architecture building ----

    def add_conv(self, fshape=(3, 3), n_filters=1, stride=1, activation=Tanh):
        fshape = [self.fanin[0]] + list(fshape)
        args = (self, fshape, self.layers[-1].outshape, n_filters, stride, len(self.layers), activation)
        self.layers.append(ConvLayer(*args))
        self.architecture.append("{}x{}x{} Conv: {}".format(fshape[0], fshape[1], n_filters, str(activation())[:4]))
        # brain, fshape, fanin, num_filters, stride, position, activation="sigmoid"

    def add_pool(self, pool=2):
        args = (self, self.layers[-1].outshape, (pool, pool), pool, len(self.layers))
        self.layers.append(PoolLayer(*args))
        self.architecture.append("{} Pool".format(pool))
        # brain, fanin, fshape, stride, position

    def add_fc(self, neurons, activation=Tanh):
        inpts = np.prod(self.layers[-1].outshape)
        args = (self, inpts, neurons, len(self.layers), activation)
        self.layers.append(FFLayer(*args))
        self.architecture.append("{} FC: {}".format(neurons, str(activation())[:4]))
        # brain, inputs, neurons, position, activation

    def add_drop(self, neurons, dropchance=0.25, activation=Tanh):
        args = (self, np.prod(self.layers[-1].outshape), neurons, dropchance, len(self.layers), activation)
        self.layers.append(DropOut(*args))
        self.architecture.append("{} Drop({}): {}".format(neurons, round(dropchance, 2), str(activation())[:4]))
        # brain, inputs, neurons, dropout, position, activation

    def add_rec(self, neurons, time_truncate=5, activation=Tanh):
        inpts = np.prod(self.layers[-1].outshape)
        args = self, inpts, neurons, time_truncate, len(self.layers), activation
        self.layers.append(RLayer(*args))
        self.architecture.append("{} RecL(time={}): {}".format(neurons, time_truncate, activation[:4]))
        # brain, inputs, neurons, time_truncate, position, activation

    def finalize_architecture(self, activation=Sigmoid):
        fanin = np.prod(self.fanin)
        pargs = (self, np.prod(self.layers[-1].outshape), self.outsize, len(self.layers), activation)
        eargs = (self, np.prod(self.layers[-1].outshape), fanin, len(self.layers), activation)
        self.predictor = FFLayer(*pargs)
        self.encoder = FFLayer(*eargs)
        self.layers.append(self.predictor)
        self.architecture.append("{}|{} Pred|Enc: {}".format(self.outsize, fanin, str(activation())[:4]))
        self.finalized = True

    # ---- Methods for model fitting ----

    def learn(self, batch_size):
        if not self.finalized:
            raise RuntimeError("Architecture not finalized!")
        if self.layers[-1] is not self.predictor:
            self._insert_predictor()
        for no, batch in enumerate(self.data.batchgen(batch_size)):
            self.m = batch[0].shape[0]
            self._fit(batch)

        self.age += 1

    def autoencode(self, batch_size):
        """
        Coordinates the autoencoding of the myData

        :param batch_size: the size of said batches
        :return None
        """
        if not self.finalized:
            raise RuntimeError("Architecture not finalized!")
        if self.layers[-1] is not self.encoder:
            self._insert_encoder()
        self.N = self.data.N
        for batch in self.data.batchgen(batch_size):
            self.m = batch[0].shape[0]
            self._fit((batch[0], rtm(batch[0])))

    def new_autoencode(self, batches, batch_size):

        def sanity_check():
            if not self.finalized:
                raise RuntimeError("Architecture not finalized!")
            if self.layers[-1] is not self.encoder:
                self._insert_encoder()

        def train_encoder():
            for no in range(batches):
                stimuli = questions[no*batch_size:(no*batch_size)+batch_size]
                self.m = stimuli.shape[0]
                target = rtm(stimuli)
                self.encoder.error = self.cost.derivative(
                    self.encoder.feedforward(stimuli),
                    target,
                    self.encoder.excitation,
                    self.encoder.activation)
                self.error = sum(np.average(self.encoder.error, axis=0))
                self.encoder.weight_update()

        def autoencode_layers():
            for layer in self.layers[1:-1]:
                autoencoder = self.layers[:1] + [layer] + [self.encoder]
                for no in range(batches):
                    batch = questions[no*batch_size:(no*batch_size)+batch_size]
                    self.m = batch.shape[0]
                    target = rtm(batch)
                    for lyr in autoencoder:
                        lyr.feedforward(batch)
                    self.encoder.error = self.cost.derivative(self.encoder.output,
                                                              target,
                                                              self.encoder.excitation,
                                                              self.encoder.activation)
                    autoencoder[-2].error = self.encoder.backpropagation()
                    self.error = sum(np.average(self.encoder.error, axis=0))
                    for lyr in autoencoder[1:]:
                        lyr.weight_update()

        sanity_check()

        questions = self.data.learning[:batches*batch_size]
        self.N = questions.shape[0]

        train_encoder()
        if len(self.layers) > 2:
            autoencode_layers()

    def _fit(self, learning_table):
        """
        This method coordinates the fitting of parameters (learning and encoding).

        Backprop and weight update is implemented layer-wise and
        could be somewhat parallelized.
        Backpropagation is done a bit unorthodoxically. Each
        layer computes the error of the previous layer and the
        backprop methods return with the computed error array

        :param learning_table: tuple of 2 numpy arrays: (stimuli, targets)
        :return: None
        """

        questions, targets = learning_table
        # Forward pass
        for layer in self.layers:
            questions = layer.feedforward(questions)
        # Calculate the cost derivative
        last = self.layers[-1]
        last.error = self.cost.derivative(last.output, targets, last.activation)
        # Backpropagate the errors
        for layer in self.layers[-1:1:-1]:
            prev_layer = self.layers[layer.position - 1]
            prev_layer.receive_error(layer.backpropagation())
        # Update weights
        for layer in self.layers[1:]:
            layer.weight_update()

        # Calculate the sum of errors in the last layer, averaged over the batch
        self.error = sum(np.average(self.layers[-1].error, axis=0))

    # ---- Methods for forward propagation ----

    def predict(self, questions):
        """
        Coordinates prediction (feedforwarding outside the learning phase)

        The layerwise implementations of <_evaluate> don't have any side-effects
        so prediction is a candidate for parallelization.

        :param questions: numpy.ndarray representing a batch of inputs
        :return: numpy.ndarray: 1D array of predictions
        """
        for layer in self.layers:
            questions = layer.predict(questions)
        if self.data.type == "classification":
            return np.argmax(questions, axis=1)
        else:
            return questions

    def evaluate(self, on="testing"):
        """
        Calculates the network's prediction accuracy.

        :param on: cross-validation is implemented, dataset can be chosen here
        :return: rate of right answers
        """

        def revaluate(preds):
            ideps = {"d": d.indeps, "l": d.lindeps, "t": d.tindeps}[on[0]][:m]
            return np.mean(np.sqrt((d.upscale(preds) - d.upscale(ideps)) ** 2))

        def cevaluate(preds):
            ideps = self.data.dummycode(on)
            return np.mean(np.equal(ideps, preds))

        m = self.data.n_testing
        d = self.data
        evalfn = revaluate if d.type == "regression" else cevaluate
        questions = {"d": d.data[:m], "l": d.learning[:m], "t": d.testing}[on[0]]
        result = evalfn(self.predict(questions))
        return result

    # ---- Private helper methods ----

    def _insert_encoder(self):
        if not isinstance(self.cost, MSE) or self.cost is not MSE:
            print("Chosen cost function not supported in autoencoding!\nAttention! Falling back to MSE!")
            self.cost = MSE()
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
        chain += "{}\n".format(name)
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
        assert all([(isinstance(layer.activation, Sigmoid)) or (layer.activation is Sigmoid)
                    for layer in self.layers[1:]]), "Only Sigmoid is supported!"

        from csxnet.utilities.nputils import logit

        print("Warning! Network.dream() is highly experimental and possibly buggy!")

        for layer in self.layers[-1:0:-1]:
            matrix = logit(matrix.dot(layer.weights.T))
        return matrix


class FFNN(Network):
    """
    Layerwise representation of a Feed Forward Neural Network

    Learning rate is given by the keyword argument <rate>.
    The neural network architecture is given by <hiddens>.
    Multiple hidden layers may be defined. Input and output neurons
    are calculated from the shape of <data>
    """

    def __init__(self, hiddens, data, eta, lmbd1=0.0, lmbd2=0.0, mu=0.0,
                 cost=Xent, activation=Tanh):
        Network.__init__(self, data=data, eta=eta, lmbd1=lmbd1, lmbd2=lmbd2, mu=mu, cost=cost)

        if isinstance(hiddens, int):
            hiddens = (hiddens,)

        self.layout = tuple(list(self.layers[0].outshape) + list(hiddens) + [self.outsize])

        for neu in hiddens:
            self.add_fc(neurons=neu, activation=activation)
        self.finalize_architecture(activation=Sigmoid)


class RNN(Network):
    def __init__(self, hiddens, data, eta, cost=Xent, activation=Tanh):
        Network.__init__(self, data, eta, 0.0, 0.0, 0.0, cost)

        if isinstance(hiddens, int):
            hiddens = (hiddens,)

        for neu in hiddens:
            self.add_rec(neu, activation=activation)

        self.finalize_architecture(activation=Sigmoid)

    def learn(self, time):
        Network.learn(self, time)


class ThNetDynamic(NeuralNetworkBase):
    def __init__(self, data, eta, lmbd1, lmbd2, mu, cost):
        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu)

        inshape, _ = data.neurons_required()

        if isinstance(inshape, int):
            inshape = (inshape,)

        self.inputs = T.matrix("Inputs") if len(inshape) == 1 else T.tensor4("Inputs")
        self.targets = T.matrix("Targets")
        self.m = T.scalar("m", dtype="float32")
        self.mint = T.scalar("mint", dtype="int32")

        # Final computational graphs go here
        self.cost = cost
        self.output = None
        self.prediction = None

        # Compiled Theno functions will go here
        self._fit = None
        self._evaluate = None
        self._predict = None

        self.age = 0
        self.architecture = []
        self.finalized = False

    def add_convpool(self, conv, filters, pool):
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("Conv({}x{}x{}); Pool({})".format(filters, conv, conv, pool))
        self.layers.append(ThConvPoolLayer(conv, filters, pool, fanin, pos))

    def add_fc(self, neurons, activation="sigmoid"):
        if activation == "softmax":
            if neurons != self.outsize:
                print("Warning! Assumed creation of output layer, but got wrong number of neurons!")
                print("Adjusting neuron number to correct output size:", self.outsize)
            self.finalize()

        pos, fanin = self._layeradd_prepare()

        self.architecture.append("FC({}): {}".format(neurons, activation[:4]))
        self.layers.append(ThFCLayer(neurons, fanin, pos, activation))

    def add_rlayer(self, neurons):
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("Rlayer({}): {}".format(neurons, "tanh"))
        self.layers.append(ThRLayer(neurons, fanin, pos))

    def add_lstm(self, neurons):
        pos, fanin = self._layeradd_prepare()

        self.architecture.append("LSTM({}): {}".format(neurons, "tanh"))
        self.layers.append(ThLSTM(neurons, fanin, pos))

    def _layeradd_prepare(self):
        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape
        return pos, fanin

    def finalize(self):

        def define_output_layer():
            pos, fanin = self._layeradd_prepare()
            self.architecture.append("Out({}): {}".format(self.outsize, "softmax"))
            return ThOutputLayer(self.outsize, fanin, pos)

        def define_feedforward():
            self.mint = self.m.astype("int32")
            forward_pass_rules = [self.inputs]
            for layer in self.layers:
                forward_pass_rules.append(layer.output(forward_pass_rules[-1], self.mint))
            return forward_pass_rules[-1]

        def define_cost():
            # Define regularization terms
            # l1 = sum([T.abs_(layer.weights).sum() for layer in self.layers])
            # l1 *= self.lmbd1 / (self.data.N / 2)
            # l2 = sum([T.exp2(layer.weights).sum() for layer in self.layers])
            # l2 *= self.lmbd2 / (self.data.N * 2)

            # Build string for architecture display
            chain = "Cost: " + self.cost
            reg = ""
            if self.lmbd1 or self.lmbd2:
                reg += " + "
            if self.lmbd1:
                reg += "L1"
                if self.lmbd2:
                    reg += " + L2 reg."
                else:
                    reg += " reg."
            if self.lmbd2:
                reg += "L2 reg."
            self.architecture.append(chain + reg)

            # Define cost function
            if self.cost.lower() == "xent":
                cost = nnet.categorical_crossentropy(
                    coding_dist=self.output,
                    true_dist=self.targets).sum()
            elif self.cost.lower() == "mse":
                cost = T.exp2(self.targets - self.output).sum()
            else:
                raise RuntimeError("Cost function {} not supported!".format(self.cost))

            # cost = cost + l1 + l2

            return cost

        def define_update_rules():
            rules = []
            for params in [layer.params for layer in self.layers]:
                rules.extend([(param, param - (self.eta / self.m) * T.grad(self.cost, param))
                              for param in params])
            return rules

        self.layers.append(define_output_layer())
        self.output = define_feedforward()
        self.cost = define_cost()
        self.prediction = T.argmax(self.output, axis=1)
        updates = define_update_rules()

        self._fit = theano.function(inputs=[self.inputs, self.targets, self.m],
                                    updates=updates,
                                    name="_fit")

        self._evaluate = theano.function(inputs=[self.inputs, self.targets, self.m],
                                         outputs=[self.cost, self.prediction],
                                         name="_evaluate")
        self._predict = theano.function(inputs=[self.inputs, self.m],
                                        outputs=[self.prediction],
                                        name="_predict")
        self.finalized = True

    def learn(self, batch_size, verbose=False):
        if not self.finalized:
            raise RuntimeError("Unfinalized network!")
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = float(questions.shape[0])
            self._fit(questions, targets, m)
            if verbose:
                print("Done {} batches!".format(i))
        self.age += 1

    def evaluate(self, on="tesing"):
        if not self.finalized:
            raise RuntimeError("Unfinalized network!")
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._evaluate(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc

    def predict(self, questions: np.ndarray):
        m = questions.shape[0]
        return self._predict(questions, m)

    def describe(self, verbose=1):
        chain = "---------------\n"
        chain += "Dynamic Theano-based Artificial Neural Network.\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + "; ".join(self.architecture) + "\n"
        if not self.finalized:
            chain += "!!! UNFINALIZED !!!"
        chain += "---------------"
        if verbose:
            print(chain)
        else:
            return chain
