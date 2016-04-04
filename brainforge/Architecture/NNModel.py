"""CNN implementation"""
import warnings

from ..Layerdef.Layers import *
from ..Utility.activations import *
from ..Utility.utility import ravel_to_matrix as rtm


class Network:
    def __init__(self, data, eta: float, lmbd: float, cost: callable):

        self.cost = cost()
        self.error = float()
        self.eta = eta
        self.lmbd = lmbd
        self.inshape, self.outsize = data.neurons_required()
        self.N = data.N  # Number of training inputs
        self.m = int()  # Batch size goes here

        self.layers = []
        self.layers.append(InputLayer(brain=self, inshape=self.inshape))
        self.predictor = None
        self.encoder = None

        self.finalized = False

        self.data = data

        self.age = 0

    # ---- Methods for architecture building ----

    def add_conv(self, fshape=(3, 3), n_filters=1, stride=1, activation=Sigmoid):
        fshape = [self.inshape[0]] + list(fshape)
        args = (self, fshape, self.layers[-1].outshape, n_filters, stride, len(self.layers), activation)
        self.layers.append(ConvLayer(*args))
        # brain, fshape, inshape, num_filters, stride, position, activation="sigmoid"

    def add_pool(self, pool=2):
        args = (self, self.layers[-1].outshape, (pool, pool), pool, len(self.layers))
        self.layers.append(PoolLayer(*args))
        # brain, inshape, fshape, stride, position

    def add_fc(self, neurons, activation=Sigmoid):
        args = (self, np.prod(self.layers[-1].outshape), neurons, len(self.layers), activation)
        self.layers.append(FFLayer(*args))
        # brain, inputs, neurons, position, activation

    def add_drop(self, neurons, dropchance=0.25, activation=Sigmoid):
        args = (self, np.prod(self.layers[-1].outshape), neurons, dropchance, len(self.layers), activation)
        self.layers.append(DropOut(*args))
        # brain, inputs, neurons, dropout, position, activation

    def finalize_architecture(self, activation=Sigmoid):
        pargs = (self, np.prod(self.layers[-1].outshape), self.outsize, len(self.layers), activation)
        eargs = (self, np.prod(self.layers[-1].outshape), np.prod(self.inshape), len(self.layers), activation)
        self.predictor = FFLayer(*pargs)
        self.encoder = FFLayer(*eargs)
        self.layers.append(self.predictor)
        self.finalized = True
        # print("--- Finalized  Architecture ---")

    # ---- Methods for model fitting ----

    def learn(self, batch_size: int):
        if not self.finalized:
            raise RuntimeError("Architecture not finalized!")
        if self.layers[-1] is not self.predictor:
            self._insert_predictor()
        for no, batch in enumerate(self.data.batchgen(batch_size)):
            self.m = batch[0].shape[0]
            self._fit(batch)

        self.age += 1

    def autoencode(self, batches: int, batch_size: int):
        """
        Coordinates the autoencoding of the myData

        :param batches: the number of batches to autoencode
        :param batch_size: the size of said batches
        :return None
        """
        if not self.finalized:
            raise RuntimeError("Architecture not finalized!")

        print("---  Starting  autoencoding  ---")
        print("On {} datapoints".format(batch_size))

        if self.layers[-1] is not self.encoder:
            self._insert_encoder()

        questions = self.data.learning[:batches*batch_size]
        targets = rtm(questions)
        self.N = questions.shape[0]
        self.m = batch_size

        for no in range(batches):
            self._fit((questions[no*batch_size:(no*batch_size)+batch_size],
                       targets[no*batch_size:(no*batch_size)+batch_size]))
            # print("Autoencoding: batch {} done! Cost' = {}".format(no+1, self.error))

        print("---  Finished  autoencoding  ---")

    def new_autoencode(self, batches: int, batch_size: int):

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
        last = self.predictor
        last.error = self.cost.derivative(last.output, targets, last.excitation, last.activation)
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

    def predict(self, questions: np.ndarray):
        """
        Coordinates prediction (feedforwarding outside the learning phase)

        The layerwise implementations of <predict> don't have any side-effects
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

    def _revaluate(self, preds, on):
        m = self.data.n_testing
        ideps = {"d": self.data.indeps, "l": self.data.lindeps, "t": self.data.tindeps}[on[0]][:m]
        return np.average(np.sqrt((preds - ideps)**2))

    def _cevaluate(self, preds, on):
        ideps = self.data.dummycode(on)
        return np.average(np.equal(ideps, preds))

    def evaluate(self, on="testing"):
        """
        Calculates the network's prediction accuracy.

        :param on: cross-validation is implemented, dataset can be chosen here
        :return: rate of right answers
        """
        evalfn = self._revaluate if self.data.type == "regression" else self._cevaluate
        m = self.data.n_testing
        questions = {"d": self.data.data[:m], "l": self.data.learning[:m], "t": self.data.testing}[on[0]]
        result = evalfn(self.predict(questions), on)
        return result

    # ---- Private helper methods ----

    def _insert_encoder(self):
        self.layers[-1] = self.encoder
        print("Inserted encoder as output layer!")

    def _insert_predictor(self):
        self.layers[-1] = self.predictor
        print("Inserted predictor as output layer!")

    # ---- Experimental section ----

    def mp_predict(self, questions):
        jobs = mp.cpu_count() + 1
        pool = mp.Pool(jobs)
        predictions = pool.map(self.predict, questions, chunksize=10)
        return np.concatenate(predictions)

    def _predwrap(self, batch, no):
        return no, self.predict(batch)
