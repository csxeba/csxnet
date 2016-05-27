"""CNN implementation"""
from .NetworkBase import NeuralNetworkBase
from ..Layerdef.Layers import *
from ..Utility.activations import *
from ..Utility.cost import *


class Network(NeuralNetworkBase):
    def __init__(self, data, eta: float, lmbd1: float, lmbd2, mu: float, cost: callable):

        NeuralNetworkBase.__init__(self, data, eta, lmbd1, lmbd2, mu, cost)

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
            self._fit((batch[0], ravtm(batch[0])))

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
                target = ravtm(stimuli)
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
                    target = ravtm(batch)
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
        from ..Utility.cost import MSE
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
        chain = "----------\n"
        chain += "CsxNet BrainForge Artificial Neural Network.\n"
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

        from csxnet.nputils import logit

        print("Warning! Network.dream() is highly experimental and possibly buggy!")

        for layer in self.layers[-1:0:-1]:
            matrix = logit(matrix.dot(layer.weights.T))
        return matrix


class FFNN(Network):
    """
    Layerwise representation of a Feed Forward Neural Network

    Learning rate is given by the keyword argument <rate>.
    The neural network architecture is given by <*layout> in the form of
    inputs, hiddens, outsize. Multiple hidden layers may be defined.
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