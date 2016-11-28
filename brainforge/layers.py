import abc
import warnings

import numpy as np

from ..util import white, white_like
from ..ops import sigmoid


class _Layer(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, activation, **kw):

        from ..ops import act_fns as activations

        self.brain = None
        self.inputs = None
        self.output = None

        self.connected = False

        self.optimizer = None

        if isinstance(activation, str):
            self.activation = activations[activation]
        else:
            self.activation = activation

        if "trainable" in kw:
            self.trainable = kw["trainable"]
        else:
            self.trainable = True

    @abc.abstractmethod
    def connect(self, to, inshape):
        self.brain = to

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: raise NotImplementedError

    @abc.abstractmethod
    def backpropagate(self, error) -> np.ndarray: raise NotImplementedError

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
            self.weights, self.biases = w

    @property
    def gradients(self):
        return np.concatenate([self.nabla_w.ravel(), self.nabla_b.ravel()])

    @property
    def nparams(self):
        return self.weights.size + self.biases.size

    @property
    @abc.abstractmethod
    def outshape(self): raise NotImplementedError

    @abc.abstractmethod
    def __str__(self): raise NotImplementedError


class _VecLayer(_Layer):
    """Base class for layer types, which operate on tensors
     and are sparsely connected"""

    @abc.abstractmethod
    def connect(self, to, inshape):
        _Layer.connect(self, to, inshape)


class _FFLayer(_Layer):
    """Base class for the fully connected layer types"""
    def __init__(self, neurons: int, activation, **kw):
        _Layer.__init__(self, activation, **kw)

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

    @property
    def outshape(self):
        return self.neurons if isinstance(self.neurons, tuple) else (self.neurons,)


class _Recurrent(_FFLayer):

    def __init__(self, neurons, activation, return_seq=False):
        _FFLayer.__init__(self, neurons, activation)
        self.Z = 0
        self.Zs = []
        self.cache = []
        self.gates = []

        self.time = 0
        self.return_seq = return_seq

        self.cache = None

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray):
        self.inputs = stimuli.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.Zs, self.gates, self.cache = [], [], []
        return np.zeros((self.brain.m, self.neurons))

    @abc.abstractmethod
    def backpropagate(self, error):
        if self.return_seq:
            return error.transpose(1, 0, 2)
        else:
            error_tensor = np.zeros((self.time, self.brain.m, self.neurons))
            error_tensor[-1] = error
            return error_tensor

    @property
    def outshape(self):
        if self.return_seq:
            return self.time, self.neurons
        else:
            return self.neurons,


class _Op(_Layer):

    def __init__(self, op):
        _Layer.__init__(self, activation="linear", trainable=False)
        self.op = op

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        self.output = self.op(stimuli)
        return self.output

    def connect(self, to, inshape):
        _Layer.connect(self, to, inshape)
        self.op.inshape = inshape

    def backpropagate(self, error) -> np.ndarray:
        return self.op.backwards(error)

    def __str__(self):
        return str(self.op)

    def get_weights(self, unfold=True):
        return NotImplemented

    def set_weights(self, w, fold=True):
        return NotImplemented

    def shuffle(self) -> None:
        return NotImplemented

    @property
    def outshape(self):
        return self.op.outshape()


class InputLayer(_Layer):

    def __init__(self, shape):
        _Layer.__init__(self, activation="linear", trainable=False)
        self.neurons = shape

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

    def __init__(self, neurons, activation, **kw):
        if isinstance(neurons, tuple):
            neurons = neurons[0]
        _FFLayer.__init__(self,  neurons=neurons, activation=activation, **kw)

    def connect(self, to, inshape):
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
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


class Flatten(_Op):

    def __init__(self):
        from ..ops import Flatten as op
        _Op.__init__(self, op=op())

    def connect(self, to, inshape):
        _Op.connect(self, to, inshape)


class DropOut(_Layer):

    def __init__(self, dropchance):
        _Layer.__init__(self, activation="linear", trainable=False)
        self.dropchance = 1. - dropchance
        self.mask = None
        self.neurons = None

    def connect(self, to, inshape):
        self.neurons = inshape

    def feedforward(self, questions):
        self.inputs = questions
        self.mask = np.random.uniform(0, 1, self.neurons) < self.dropchance
        self.output = questions * self.mask
        return self.output

    def backpropagate(self, error):
        return error * self.mask

    def get_weights(self, unfold=True): raise NotImplementedError

    def set_weights(self, w, fold=True): raise NotImplementedError

    def shuffle(self) -> None: raise NotImplementedError

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

        output = _Recurrent.feedforward(self, questions)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=-1)
            output = self.activation(Z.dot(self.weights) + self.biases)

            self.Zs.append(Z)
            self.cache.append(output)

        if self.return_seq:
            self.output = np.stack(self.cache, axis=1)
        else:
            self.output = self.cache[-1]

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
        deltaX = np.zeros_like(self.inputs)

        for time in range(self.time-1, -1, -1):
            output = self.cache[time]
            Z = self.Zs[time]

            delta += error[time]
            delta *= self.activation.derivative(output)

            self.nabla_w += Z.T.dot(delta)
            self.nabla_b += delta.sum(axis=0)

            deltaZ = delta.dot(self.weights.T)
            deltaX[time] = deltaZ[:, :-self.neurons]
            delta = deltaZ[:, -self.neurons:]

        return deltaX.transpose(1, 0, 2)

    def __str__(self):
        return "{}-RLayer-{}".format(self.neurons, str(self.activation))


class LSTM(_Recurrent):

    def __init__(self, neurons, activation, return_seq=False):
        _Recurrent.__init__(self, neurons, activation, return_seq)
        self.G = neurons * 3
        self.Zs = []
        self.gates = []

    def connect(self, to, inshape):
        _Recurrent.connect(self, to, inshape)
        self.Z = inshape[-1] + self.neurons
        self.weights = white(self.Z, self.neurons * 4)
        self.biases = np.zeros((self.neurons * 4,))

    def feedforward(self, X: np.ndarray):

        output = _Recurrent.feedforward(self, X)
        state = np.zeros_like(output)

        for t in range(self.time):
            Z = np.concatenate((self.inputs[t], output), axis=1)

            preact = Z.dot(self.weights) + self.biases
            preact[:, :self.G] = sigmoid(preact[:, :self.G])
            preact[:, self.G:] = self.activation(preact[:, self.G:])

            f, i, o, cand = np.split(preact, 4, axis=-1)

            state = state * f + i * cand
            state_a = self.activation(state)
            output = state_a * o

            self.Zs.append(Z)
            self.gates.append(preact)
            self.cache.append([output, state_a, state, preact])

        if self.return_seq:
            self.output = np.stack([cache[0] for cache in self.cache], axis=1)
        else:
            self.output = self.cache[-1][0]
        return self.output

    def backpropagate(self, error):

        error = _Recurrent.backpropagate(self, error)

        self.nabla_w = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)

        actprime = self.activation.derivative
        sigprime = sigmoid.derivative

        dstate = np.zeros_like(error[-1])
        deltaX = np.zeros_like(self.inputs)
        deltaZ = np.zeros_like(self.Zs[0])

        for t in range(-1, -(self.time+1), -1):
            output, state_a, state, preact = self.cache[t]
            f, i, o, cand = np.split(self.gates[t], 4, axis=-1)

            # Add recurrent delta to output delta
            error[t] += deltaZ[:, -self.neurons:]

            # Backprop into state
            dstate += error[t] * o * actprime(state_a)

            state_yesterday = 0. if t == -self.time else self.cache[t-1][2]
            # Calculate the gate derivatives
            dfgate = state_yesterday * dstate
            digate = cand * dstate
            dogate = state_a * error[t]
            dcand = i * dstate * actprime(cand)  # Backprop nonlinearity
            dgates = np.concatenate((dfgate, digate, dogate, dcand), axis=-1)
            dgates[:, :self.G] *= sigprime(self.gates[t][:, :self.G])  # Backprop nonlinearity

            dstate *= f

            self.nabla_b += dgates.sum(axis=0)
            self.nabla_w += self.Zs[t].T.dot(dgates)

            deltaZ = dgates.dot(self.weights.T)

            deltaX[t] = deltaZ[:, :-self.neurons]

        return deltaX.transpose(1, 0, 2)

    def __str__(self):
        return "{}-LSTM-{}".format(self.neurons, str(self.activation)[:4])


class EchoLayer(RLayer):

    def __init__(self, neurons, activation, return_seq=False, p=0.1):
        RLayer.__init__(self, neurons, activation, return_seq)
        self.trainable = False
        self.p = p

    def connect(self, to, inshape):
        RLayer.connect(self, to, inshape)
        self.weights = np.random.binomial(1., self.p, size=self.weights.shape).astype(float)
        self.weights *= np.random.randn(*self.weights.shape)
        self.biases = white_like(self.biases)

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

        def __init__(self, nfilters, filterx, filtery, activation="relu", **kw):

            _VecLayer.__init__(self, activation=activation, **kw)

            self.nfilters = nfilters
            self.fx = filterx
            self.fy = filtery
            self.depth = 0
            self.stride = 1

            self.inshape = None

            self.error = None
            self.weights = None
            self.biases = None
            self.nabla_w = None
            self.nabla_b = None
            self.op = None

        def connect(self, to, inshape):
            from ..ops import ScipySigConv

            _VecLayer.connect(self, to, inshape)
            depth, ix, iy = inshape
            self.op = ScipySigConv(filtershape=(self.fx, self.fy), mode="valid")
            self.inshape = inshape
            self.depth = depth
            self.weights = white(self.fx, self.fy, self.depth, self.nfilters)
            self.biases = np.zeros((self.nfilters,))

        def feedforward(self, X):
            self.inputs = np.copy(X)
            out = self.activation(self.op(X, self.weights))
            self.output = out.reshape(self.brain.m, *self.outshape)
            return self.output

        def backpropagate(self, error):
            """

            :param error: 4D tensor: (m, filter_number, x, y)
            :return:
            """

            self.error = error * self.activation.derivative(self.output)
            # iT = np.rot90(self.inputs.transpose(2, 3, 1, 0), 2)
            # iT = iT.transpose(2, 3, 0, 1)
            iT = self.inputs.transpose(1, 0, 2, 3)
            eT = self.error.transpose(2, 3, 0, 1)
            self.nabla_w = self.op(iT, eT).transpose(2, 3, 1, 0)
            self.nabla_b = self.error.sum(axis=(0, 2, 3))
            return self.op.backwards(self.error, self.weights).reshape(self.inputs.shape)

        @property
        def outshape(self):
            ox, oy = tuple(ix - fx + 1 for ix, fx in zip(self.inshape[-2:], (self.fx, self.fy)))
            return self.nfilters, ox, oy

        def __str__(self):
            return "Conv({}x{}x{})-{}".format(self.nfilters, self.fx, self.fy, str(self.activation)[:4])

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
