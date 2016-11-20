import abc
import warnings

import numpy as np

from ..util import white, white_like, sigmoid


class _Layer(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, activation):

        from ..util import act_fns as activations

        self.brain = None
        self.inputs = None
        self.output = None

        self.trainable = True
        self.connected = False

        self.optimizer = None

        if isinstance(activation, str):
            self.activation = activations[activation]
        else:
            self.activation = activation

    @abc.abstractmethod
    def connect(self, to, inshape):
        self.brain = to

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: raise NotImplemented

    @abc.abstractmethod
    def backpropagate(self, error) -> np.ndarray: raise NotImplemented

    @abc.abstractmethod
    def shuffle(self) -> None: raise NotImplemented

    @abc.abstractmethod
    def get_weights(self, unfold=True): raise NotImplemented

    @abc.abstractmethod
    def set_weights(self, w, fold=True): raise NotImplemented

    @property
    @abc.abstractmethod
    def outshape(self): raise NotImplemented

    @property
    @abc.abstractmethod
    def __str__(self): raise NotImplemented


class _VecLayer(_Layer):
    """Base class for layer types, which operate on tensors
     and are sparsely connected"""
    def __init__(self, fshape: tuple, stride: int, activation):
        _Layer.__init__(self, activation)

        if len(fshape) != 3:
            fshape = (None, fshape[0], fshape[1])

        self.fshape = fshape
        self.stride = stride

    @abc.abstractmethod
    def connect(self, to, inshape): raise NotImplemented


class _FFLayer(_Layer):
    """Base class for the fully connected layer types"""
    def __init__(self, neurons: int, activation):
        _Layer.__init__(self, activation)

        self.neurons = neurons

        self.weights = None
        self.biases = None
        self.nabla_w = None
        self.nabla_b = None

    @abc.abstractmethod
    def connect(self, to, inshape):
        _Layer.connect(self, to, inshape)
        self.nabla_w = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)

    def shuffle(self) -> None:
        self.weights = white_like(self.weights)
        self.biases = np.zeros_like(self.biases)

    def get_weights(self, unfold=True):
        if unfold:
            return np.concatenate((self.weights.ravel(), self.biases.ravel()))
        return [self.weights, self.biases]

    def set_weights(self, w, fold=True):
        if fold:
            sw = self.weights
            self.weights = w[:sw.size].reshape(sw.shape)
            self.biases = w[sw.size:].reshape(self.biases.shape)
        else:
            self.weights = w[0]
            self.biases = w[1]

    @property
    def gradients(self):
        return np.concatenate([self.nabla_w.ravel(), self.nabla_b.ravel()])

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)

    @property
    def nparams(self):
        return self.weights.size + self.biases.size


class _Recurrent(_FFLayer):

    def __init__(self, neurons, activation, return_seq=False):
        _FFLayer.__init__(self, neurons, activation)
        self.Z = 0

        self.time = 0
        self.return_seq = return_seq

        self.cache = None

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray):
        self.inputs = stimuli.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.cache.reset(batch_size=self.brain.m, time=self.time)
        return np.zeros((self.brain.m, self.neurons))

    @abc.abstractmethod
    def backpropagate(self, error):
        if self.return_seq:
            return error.transpose(1, 0, 2)
        else:
            error_tensor = np.zeros((self.time, self.brain.m, self.neurons))
            error_tensor[-1] = error
            return error_tensor

    @abc.abstractmethod
    def connect(self, to, inshape):
        _FFLayer.connect(self, to, inshape)
        self.cache = self.Cache(inshape[-1], self.neurons)

    @property
    def outshape(self):
        if self.return_seq:
            return self.time, self.neurons
        else:
            return self.neurons,

    class Cache:

        def __init__(self, inp, neu):
            self.keys = ("outputs", "tanh states", "gate output", "states",
                         "candidates", "gate input", "gate forget", "Z")
            self.k = len(self.keys)
            self.Z = inp + neu
            self.n = neu
            self.m = None
            self.t = None
            self.innards = None
            self.Zs = None

        def assertkey(self, item):
            if item not in self.keys:
                raise IndexError("There is no cache named {}".format(item))

        def reset(self, batch_size=None, time=None):
            if self.t is None and time is None:
                raise RuntimeError("No previous time information is available! Supply time parameter!")
            if time is None:
                time = self.t
            if self.m is None and batch_size is None:
                raise RuntimeError("No previous time information is available! Supply time parameter!")
            if batch_size is None:
                batch_size = self.m

            self.t = time
            self.m = batch_size
            self.innards = np.zeros((self.k, time, self.m, self.n))
            self.Zs = np.zeros((self.t, self.m, self.Z))

        def set_z(self, value):
            assert value.shape == self.Zs.shape
            self.Zs = value

        def __getitem__(self, item):
            self.assertkey(item)
            if item == "Z":
                return self.Zs
            return self.innards[self.keys.index(item)]

        def __setitem__(self, key, value):
            self.assertkey(key)
            if value == "Z":
                self.set_z(value)
                return
            if value.shape != (self.m, self.n):
                raise ValueError("Shapes differ: self: {} != {} :value"
                                 .format((self.m, self.n), value.shape))
            self.innards[self.keys.index(key)] = value

        def __delitem__(self, key):
            self.assertkey(key)
            if key == "Z":
                self.Zs = np.zeros((self.t, self.m, self.Z))
            self[key] = np.zeros((self.m, self.n))


class InputLayer(_Layer):

    def __init__(self, shape):
        _Layer.__init__(self, activation="linear")
        self.neurons = shape
        self.trainable = False

    def connect(self, to, inshape):
        _Layer.connect(self, to, inshape)
        assert inshape == self.neurons

    def feedforward(self, questions):
        """
        Passes the unmodified input matrix

        :param questions: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = self.output = questions
        return questions

    def backpropagate(self, error): pass

    def shuffle(self): pass

    def get_weights(self, unfold=True):
        return None

    def set_weights(self, w, fold=True):
        pass

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)

    @property
    def weights(self):
        warnings.warn("Queried weights of an InputLayer!", RuntimeWarning)
        return None

    def __str__(self):
        return str(self.neurons)


class DenseLayer(_FFLayer):
    """Just your regular Densely Connected Layer

    Aka Dense (Keras), Fully Connected, Feedforward, etc.
    Elementary building block of the Multilayer Perceptron.
    """

    def __init__(self, neurons, activation):
        if isinstance(neurons, tuple):
            neurons = neurons[0]
        _FFLayer.__init__(self,  neurons=neurons, activation=activation)

    def connect(self, to, inshape):
        assert len(inshape) == 1
        self.weights = white(inshape[0], self.neurons)
        self.biases = np.zeros((self.neurons,))
        _FFLayer.connect(self, to, inshape)

    def feedforward(self, questions):
        """
        Transforms the input matrix with a weight matrix.

        :param questions: numpy.ndarray of shape (lessons, prev_layer_output)
        :return: numpy.ndarray: transformed matrix
        """
        self.inputs = questions
        self.output = self.activation(np.dot(questions, self.weights) + self.biases)
        return self.output

    def backpropagate(self, error):
        """
        Backpropagates the errors.
        Calculates gradients of the weights, then
        returns the previous layer's error.

        :param error:
        :return: numpy.ndarray
        """
        error *= self.activation.derivative(self.output)
        self.nabla_w = np.dot(self.inputs.T, error)
        self.nabla_b = np.sum(error, axis=0)
        return np.dot(error, self.weights.T)

    def __str__(self):
        return "{}-Dense-{}".format(self.neurons, str(self.activation)[:5])


class DropOut(_Layer):

    def __init__(self, dropchance):
        _Layer.__init__(self, activation="linear")
        self.dropchance = 1. - dropchance
        self.mask = None
        self.neurons = None
        self.trainable = False

    def connect(self, to, inshape):
        self.neurons = inshape

    def feedforward(self, questions):
        self.inputs = questions
        self.mask = np.random.uniform(0, 1, self.neurons) < self.dropchance
        self.output = questions * self.mask
        return self.output

    def backpropagate(self, error):
        return error * self.mask

    def get_weights(self, unfold=True): raise NotImplemented

    def set_weights(self, w, fold=True): raise NotImplemented

    def shuffle(self) -> None: raise NotImplemented

    @property
    def outshape(self):
        return self.neurons

    def __str__(self):
        return "{}-DropOut({})".format(self.neurons, self.dropchance)


class RLayer(_Recurrent):

    def connect(self, to, inshape):
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons)
        self.biases = np.zeros((self.neurons,))

        _Recurrent.connect(self, to, inshape)

    def feedforward(self, questions: np.ndarray):

        def timestep(Z):
            return self.activation(Z.dot(self.weights) + self.biases)

        output = _Recurrent.feedforward(self, questions)
        for t in range(self.time):
            concatenated_inputs = np.concatenate((self.inputs[t], output), axis=1)
            output = timestep(concatenated_inputs)

            self.cache["Z"][t] = concatenated_inputs
            self.cache["outputs"][t] = output

        if self.return_seq:
            self.output = self.cache["outputs"].transpose(1, 0, 2)
        else:
            self.output = self.cache["outputs"][-1]

        return self.output

    def backpropagate(self, error):
        """
        Backpropagation through time (BPTT)

        :param error: the deltas flowing from the next layer
        """

        error = _Recurrent.backpropagate(self, error)

        # gradient of the cost wrt the weights: dC/dW
        self.nabla_w = np.zeros_like(self.weights)
        # gradient of the cost wrt to biases: dC/db
        self.nabla_b = np.zeros_like(self.biases)
        # the gradient flowing backwards in time
        delta = np.zeros_like(error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        delta_X = np.zeros_like(self.inputs)

        for time in range(self.time-1, -1, -1):
            delta += error[time]
            delta *= self.activation.derivative(self.cache["outputs"][time])

            self.nabla_w += self.cache["Z"][time].T.dot(delta)
            self.nabla_b += delta.sum(axis=0)

            delta_Z = delta.dot(self.weights.T)
            delta_X[time] = delta_Z[:, :-self.neurons]
            delta = delta_Z[:, -self.neurons:]

        return delta_X.transpose(1, 0, 2)

    def __str__(self):
        return "{}-RLayer-{}".format(self.neurons, str(self.activation))


class LSTM(_Recurrent):

    def connect(self, to, inshape):
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons * 4)
        self.biases = np.zeros((self.neurons * 4,))
        _Recurrent.connect(self, to, inshape)

    def feedforward(self, X: np.ndarray):

        def timestep(Z, C):
            preact = Z.dot(self.weights) + self.biases
            f, i, o = np.split(sigmoid(preact[:, sum(G)]), G, axis=1)
            cand = self.activation(preact[:, sum(G):])
            C = C * f + i * cand
            thC = self.activation(C)
            h = thC * o
            return h, C, (thC, f, i, o, cand)

        G = self.neurons, self.neurons*2
        output = _Recurrent.feedforward(self, X)
        state = np.zeros((self.brain.m, self.neurons))

        for t in range(self.time):
            concatenated_inputs = np.concatenate((self.inputs[t], output), axis=1)
            output, state, cache = timestep(concatenated_inputs, state)

            self.cache["outputs"][t] = output
            self.cache["states"][t] = state
            self.cache["tanh states"][t] = cache[0]
            self.cache["gate forget"][t] = cache[1]
            self.cache["gate input"][t] = cache[2]
            self.cache["gate output"][t] = cache[3]
            self.cache["candidates"][t] = cache[4]
            self.cache["Z"][t] = concatenated_inputs

        if self.return_seq:
            self.output = self.cache["outputs"].transpose(1, 0, 2)
            return self.output
        else:
            self.output = self.cache["outputs"][-1]
            return self.output

    def backpropagate(self, error):

        error = _Recurrent.backpropagate(error)

        def bptt_timestep(t, dy, dC):
            assert dC is 0 and t == self.time
            cch = self.cache
            dC = self.activation.derivative(cch["states"][t]) * cch["gate output"][t] * dy + dC
            do = sigmoid.derivative(cch["gate output"][t]) * cch["tanh states"] * dC
            di = sigmoid.derivative(cch["gate input"][t]) * cch["candidates"] * dC
            df = sigmoid.derivative(cch["gate forget"][t]) * cch["states"][t-1] * dC
            dcand = self.activation.derivative(cch["cadidates"][t]) * cch["gate input"][t] * dC
            deltas = np.concatenate((df, di, df, do, dcand), axis=-1)
            gb = deltas.sum(axis=0)
            dZ = deltas.dot(self.weights.T)
            gW = cch["Z"][t].T.dot(deltas)
            return gW, gb, dZ, dC

        self.nabla_w = np.zeros_like(self.weights)
        dstate = np.zeros_like(self.outshape)  # so bptt dC receives + 0 @ time == self.time
        delta = error[-1]
        deltaX = np.zeros_like(self.inputs)

        for time in range(self.time, -1, -1):
            if time < self.time:
                dstate *= self.cache["gate forget"][time+1]
            delta += error[time]
            gradW, gradb, deltaZ, dstate = bptt_timestep(time, delta, dstate)
            error = deltaZ[self.neurons:]
            deltaX[time] = deltaZ[self.neurons:]
            self.nabla_w += gradW
            self.nabla_b += gradb

        return deltaX

    def __str__(self):
        return "{}-LSTM-{}".format(self.neurons, str(self.activation)[:4])


class EchoLayer(RLayer):

    def __init__(self, neurons, activation, return_seq=False, p=0.1):
        RLayer.__init__(self, neurons, activation, return_seq)
        self.p = p

    def connect(self, to, inshape):
        RLayer.connect(self, to, inshape)
        self.weights = np.random.binomial(1., self.p, size=self.weights.shape).astype(float)
        self.weights *= np.random.randn(*self.weights.shape)
        self.biases = white_like(self.biases)
        self.trainable = False

    def __str__(self):
        return "{}-Echo-{}".format(self.neurons, str(self.activation)[:4])


class Experimental:

    class PoolLayer(_VecLayer):
        def __init__(self, brain, inshape, fshape, stride, position):
            _VecLayer.__init__(self, brain=brain,
                               inshape=inshape, fshape=fshape,
                               stride=stride, position=position,
                               activation="linear")
            self.outdim = self.inshape[0], self.outshape[0], self.outshape[1]
            self.backpass_filter = None
            print("<PoolLayer> created with fanin {} and outshape {} @ position {}"
                  .format(self.inshape, self.outshape, position))

        def feedforward(self, questions):
            """
            Implementation of a max pooling layer.

            :param questions: numpy.ndarray, a batch of outsize from the previous layer
            :return: numpy.ndarray, max pooled batch
            """

            m, f = questions.shape[:2]
            result = np.zeros((m * f * len(self.coords),))
            self.backpass_filter = np.zeros_like(questions)

            index = 0
            for i in range(m):  # ith lesson of m questions
                for j in range(f):  # jth filter of f filters (-> depth of input)
                    sheet = questions[i, j]
                    for start0, end0, start1, end1 in self.coords:
                        recfield = sheet[start0:end0, start1:end1]
                        result[index] = maxpool(recfield)
                        bpf = np.equal(recfield, result[index]).astype(int)
                        # A poem about the necessity of the sum(bpf) term
                        # If the e.g. 2x2 receptive field has multiple elements with the same value,
                        # e.g. four 0.5s, then the error factor associated with the respective recfield
                        # gets counted in 4 times, because the backpass filter has 4 1s in it.
                        # I'll just scale the values back by the sum of 1s in the
                        # backpass filter, thus averaging them over the maxes in the receptive field.
                        self.backpass_filter[i, j, start0:end0, start1:end1] = bpf / np.sum(bpf)
                        index += 1

            self.output = result.reshape([m] + list(self.outshape))
            return self.output

        def predict(self, questions):
            """
            This method has no side-effects
            :param questions:
            :return:
            """
            m, f = questions.shape[:2]
            result = np.zeros((m * f * len(self.coords),))
            index = 0
            for i in range(m):
                for j in range(f):
                    sheet = questions[i, j]
                    for start0, end0, start1, end1 in self.coords:
                        recfield = sheet[start0:end0, start1:end1]
                        result[index] = maxpool(recfield)
                        index += 1
            return result.reshape([m] + list(self.outshape))

        def backpropagate(self, error):
            """
            Calculates the error of the previous layer.
            :param error:
            :return: numpy.ndarray, the errors of the previous layer
            """
            deltas = np.zeros([self.brain.m] + list(self.inshape))
            for l in range(self.brain.m):
                for z in range(self.inshape[0]):
                    vec = self.error[l, z]
                    for i, (start0, end0, start1, end1) in enumerate(self.coords):
                        bpfilter = self.backpass_filter[l, z, start0:end0, start1:end1]
                        deltas[l, z, start0:end0, start1:end1] += bpfilter * vec[i]

            prev = self.brain.layers[self.position - 1]
            return deltas * prev.activation.derivative(prev.output)

        def receive_error(self, error_matrix):
            """
            Folds the received error matrix.
            :param error_matrix: backpropagated errors from the next layer
            :return: None
            """
            self.error = error_matrix.reshape([self.brain.m] + [self.outshape[0]] +
                                              [np.prod(self.outshape[1:])])

        def weight_update(self):
            pass

        def shuffle(self):
            pass

        @property
        def outshape(self):
            return self.outdim

    class ConvLayer(_VecLayer):
        def __init__(self, brain, fshape, inshape, num_filters, stride, position, activation):
            _VecLayer.__init__(self, brain=brain,
                               inshape=inshape, fshape=fshape,
                               stride=stride, position=position,
                               activation=activation)

            chain = """TODO: fix convolution. Figure out backprop. Unify backprop and weight update. (?)"""
            print(chain)
            self.inputs = np.zeros(self.inshape)
            self.outdim = num_filters, self.outshape[0], self.outshape[1]
            self.filters = white(num_filters, np.prod(fshape))
            self.gradients = np.zeros_like(self.filters)
            self.velocity = np.zeros_like(self.filters)
            print("<ConvLayer> created with fanin {} and outshape {} @ position {}"
                  .format(self.inshape, self.outshape, position))

            from scipy import convolve
            self.convop = convolve

        def feedforward(self, questions):
            self.inputs = questions
            exc = convolve(self.inputs, self.filters, mode="valid")
            self.output = self.activation(exc)
            return self.output

        def old_feedforward(self, questions: np.ndarray):
            """
            Convolves the inputs with filters. Used in the learning phase

            :param questions: numpy.ndarray, a batch of inputs. Shape should be (lessons, channels, x, y)
            :return: numpy.ndarray: outsize convolved with filters. Shape should be (lessons, filters, cx, cy)
            """
            self.inputs = questions

            # TODO: rethink this! Not working when channel > 1.
            recfields = np.array([[np.ravel(questions[qstn][:, start0:end0, start1:end1])
                                   for start0, end0, start1, end1 in self.coords]
                                  for qstn in range(questions.shape[0])])

            osh = [self.brain.m] + list(self.outshape)
            exc = np.matmul(recfields, self.filters.T)
            exc = np.transpose(exc, (0, 2, 1)).reshape(osh)
            self.output = self.activation(exc)
            return self.output

        def predict(self, questions: np.ndarray):
            """
            Convolves the inputs with filters.

            Used in prediction and testing. This method has no side-effects and could be used
            in multiprocessing. (Hopes die last)

            :param questions: 4D tensor of shape (lessons, channels, x, y)
            :return: 4D tensor of shape (lessons, filters, cx, cy)
            """
            recfields = np.array([[np.ravel(questions[qstn][:, start0:end0, start1:end1])
                                   for start0, end0, start1, end1 in self.coords]
                                  for qstn in range(questions.shape[0])])
            osh = [questions.shape[0]] + list(self.outshape)
            return self.activation(np.transpose(np.inner(recfields, self.filters), axes=(0, 2, 1))).reshape(*osh)

        def backpropagate(self, error):
            self.gradients = convolve(np.rot90(self.error, k=2), self.inputs)
            return convolve(self.error, np.rot90(self.filters, k=2))

        def old_backpropagation(self):
            """
            Calculates the error of the previous layer.

            :return: numpy.ndarray
            """
            if self.position == 1:
                print("Warning! Backpropagating into the input layer. Bad learning method design?")
            deltas = np.zeros_like(self.inputs)

            for n in range(self.error.shape[0]):  # -> a batch element in the input 4D-tensor
                for fnum in range(self.filters.shape[0]):  # fnum -> a filter
                    errvec = np.ravel(self.error[n, fnum, ...])  # an error sheet flattened, corresponding to a filter
                    for index, (start0, end0, start1, end1) in enumerate(self.coords):
                        diff = errvec[index] * self.filters[fnum].reshape(self.fshape)
                        np.add(deltas[..., start0:end0, start1:end1], diff,
                               out=deltas[..., start0:end0, start1:end1])
            return deltas

        def receive_error(self, error_matrix: np.ndarray):
            """
            Fold the received error matrix.

            :param error_matrix: numpy.ndarray: backpropagated errors
            :return: None
            """
            self.error = error_matrix.reshape([self.brain.m] + list(self.outshape))

        def weight_update(self):
            if self.brain.lmbd2:
                l2 = l2term(self.brain.eta, self.brain.lmbd1, self.brain.N)
                self.filters *= l2
            if self.brain.lmbd1:
                l1 = l1term(self.brain.eta, self.brain.lmbd1, self.brain.N)
                self.filters -= l1 * np.sign(self.filters)

            np.subtract(self.filters, self.velocity * self.brain.mu + self.gradients, out=self.filters)
            if self.brain.mu:
                self.velocity += self.gradients

        def old_weight_update(self):
            """
            Updates convolutional filter weights with the calculated gradients.

            :return: None
            """
            f, c = self.error.shape[1], self.inputs.shape[1]
            delta = np.zeros([f] + list(self.fshape))
            for l in range(self.brain.m):  # lth of m lessons
                for i in range(f):  # ith filter of f filters
                    for j in range(c):  # jth input channel of c channels
                        # Every channel in the input gets convolved with the error matrix of the filter
                        # these matrices are then summed elementwise.
                        cvm = convolve(self.inputs[l][j], self.error[l][i], mode="same")
                        # cvm = sigconvnd(self.inputs[l][j], self.error[l][i], mode="valid")
                        # eq = np.equal(cvm, cvm_old)
                        delta[i, j, ...] += cvm  # Averaging over lessons in the batch

            # L2 regularization aka weight decay
            l2 = l2term(self.brain.eta, self.brain.lmbd, self.brain.N)
            # Update regularized weights with averaged errors
            np.add(self.filters * l2, (self.brain.eta / self.brain.m) * delta.reshape(self.filters.shape),
                   out=self.filters)

        def _getrfields(self, slices: np.ndarray):
            return np.array([[np.ravel(stim[:, start0:end0, start1:end1])
                              for start0, end0, start1, end1 in self.coords]
                             for stim in slices])

        def shuffle(self):
            self.filters = np.random.randn(*self.filters.shape) / np.sqrt(np.prod(self.inshape))

        @property
        def outshape(self):
            return self.outdim

    class AboLayer(_Layer):
        def __init__(self, brain, position, activation):
            _Layer.__init__(self, brain, position, activation)
            self.brain = brain
            self.fanin = brain.layers[-1].fanout
            self.neurons = []

        def add_minion(self, empty_network):
            minion = empty_network
            minion.add_fc(10)
            minion.finalize_architecture()
            self.neurons.append(minion)

        def feedforward(self, inputs):
            """this ain't so simple after all O.O"""
            pass

        def receive_error(self, error_vector: np.ndarray) -> None:
            pass

        def shuffle(self) -> None:
            pass

        def backpropagate(self, error) -> np.ndarray:
            pass

        def weight_update(self) -> None:
            pass

        def predict(self, stimuli: np.ndarray) -> np.ndarray:
            pass

        def outshape(self):
            return ...
