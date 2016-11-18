import abc
import warnings

import numpy as np

from ..util import act_fns
from ..util import sigmoid, tanh
from ..util import l1term, l2term, outshape, calcsteps, white, white_like

from csxdata import floatX
from csxdata.utilities.vectorops import maxpool, ravel_to_matrix as rtm


class _LayerBase(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, brain, position, activation):
        self.brain = brain
        self.position = position
        self.inputs = None
        self.trainable = True
        if isinstance(activation, str):
            self.activation = act_fns[activation]
        else:
            self.activation = activation

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def predict(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def backpropagation(self) -> np.ndarray: pass

    @abc.abstractmethod
    def weight_update(self) -> None: pass

    @abc.abstractmethod
    def receive_error(self, error_vector: np.ndarray) -> None: pass

    @abc.abstractmethod
    def shuffle(self) -> None: pass

    @abc.abstractmethod
    def get_weights(self, unfold=True): raise NotImplemented

    @abc.abstractmethod
    def set_weights(self, w, fold=True): raise NotImplemented

    @abc.abstractproperty
    def outshape(self): pass


class _VecLayer(_LayerBase):
    """Base class for layer types, which operate on tensors
     and are sparsely connected"""
    def __init__(self, brain, inshape: tuple, fshape: tuple,
                 stride: int, position: int,
                 activation):
        _LayerBase.__init__(self, brain, position, activation)

        if len(fshape) != 3:
            fshape = ("NaN", fshape[0], fshape[1])

        self.fshape, self.stride = fshape, stride

        self.inshape = inshape
        self.outshape = outshape(inshape, fshape, stride)

        self.inputs = np.zeros(inshape)
        self.output = np.zeros(self.outshape)
        self.error = np.zeros(self.outshape)

        self.coords = calcsteps(inshape, fshape, stride)

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def predict(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def backpropagation(self) -> np.ndarray: pass

    @abc.abstractmethod
    def weight_update(self) -> None: pass

    @abc.abstractmethod
    def receive_error(self, error_vector: np.ndarray) -> None: pass

    @abc.abstractmethod
    def shuffle(self) -> None: pass

    @abc.abstractproperty
    def outshape(self): pass

    @abc.abstractmethod
    def get_weights(self, unfold=True): raise NotImplemented

    @abc.abstractmethod
    def set_weights(self, w, fold=True): raise NotImplemented


class _FFLayer(_LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, brain, neurons: int, position: int, activation):
        _LayerBase.__init__(self, brain, position, activation)

        self.neurons = neurons
        self.inputs = None

        self.output = np.zeros((neurons,))
        self.error = np.zeros((neurons,))

        self.weights = None
        self.biases = None
        self.gradients = None
        self.velocity = None

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def predict(self, stimuli: np.ndarray) -> np.ndarray: pass

    @abc.abstractmethod
    def backpropagation(self) -> np.ndarray: pass

    @abc.abstractmethod
    def weight_update(self) -> None: pass

    @abc.abstractmethod
    def receive_error(self, error_vector: np.ndarray) -> None: pass

    def shuffle(self) -> None:
        self.weights = white_like(self.weights)
        self.biases = np.zeros_like(self.biases)

    def get_weights(self, unfold=True):
        return self.weights.ravel() if unfold else self.weights

    def set_weights(self, w, fold=True):
        self.weights = w.reshape(self.weights.shape) if fold else w

    @property
    def outshape(self):
        return self.neurons


class DenseLayer(_FFLayer):
    """Just your regular Densely Connected Layer

    Aka Dense (Keras), Fully Connected, Feedforward, etc.
    Elementary building block of the Multilayer Perceptron.
    """
    def __init__(self, brain, inputs, neurons, position, activation):
        _FFLayer.__init__(self, brain=brain,
                          neurons=neurons, position=position,
                          activation=activation)

        self.weights = white(int(inputs), int(neurons))
        self.biases = np.zeros((1, neurons), dtype=float)

    def feedforward(self, questions):
        """
        Transforms the input matrix with a weight matrix.

        :param questions: numpy.ndarray of shape (lessons, prev_layer_output)
        :return: numpy.ndarray: transformed matrix
        """
        self.inputs = rtm(questions)
        self.output = self.predict(questions)
        return self.output

    def predict(self, questions):
        """
        Tranfsorms an input with the weights.

        This method has no side-effects. Used in prediction and testing.

        :param questions:
        :return:
        """
        return self.activation(np.dot(rtm(questions), self.weights) + self.biases)

    def backpropagation(self):
        """
        Backpropagates the errors.
        Calculates gradients of the weights, then
        returns the previous layer's error.

        :return: numpy.ndarray
        """
        if self.brain.mu:
            self.velocity *= self.brain.mu
            self.velocity += self.gradients * (self.brain.eta / self.brain.m)
        # (dC / dW) = error * (dZ / dW), where error = (dMSE / dA) * (dA / dZ)
        self.gradients = np.dot(self.inputs.T, self.error)
        # (dC / dX) = error * (dZ / dA_), where A_ is the previous layer's output
        return np.dot(self.error, self.weights.T)

    def weight_update(self):
        """
        Performs Stochastic Gradient Descent by subtracting a portion of the
        calculated gradients from the weights and biases.

        :return: None
        """

        def apply_weight_decay():
            if self.brain.lmbd2:
                l2 = l2term(self.brain.eta, self.brain.lmbd2, self.brain.N)
                self.weights *= l2
            if self.brain.lmbd1:
                l1 = l1term(self.brain.eta, self.brain.lmbd1, self.brain.N)
                self.weights -= l1 * np.sign(self.weights)

        def descend_on_velocity():
            np.subtract(self.weights, self.velocity + self.gradients * eta,
                        out=self.weights)

        def descend_on_gradient():
            np.subtract(self.weights, self.gradients * eta,
                        out=self.weights)

        def modify_biases():
            np.subtract(self.biases, np.mean(self.error, axis=0) * eta,
                        out=self.biases)

        eta = self.brain.eta / self.brain.m
        apply_weight_decay()
        if self.brain.mu:
            descend_on_velocity()
        else:
            descend_on_gradient()
        modify_biases()

    def receive_error(self, error):
        """
        Saves the received error matrix.

        The received matrix should not be folded, since FFLayer should only be
        followed by FFLayer.

        :param error: numpy.ndarray: 2D matrix of errors
        :return: None
        """
        self.error = rtm(error) * self.activation.derivative(self.output)

    def get_weights(self, unfold=True):
        return self.weights.ravel() if unfold else self.weights

    def set_weights(self, w, fold=True):
        self.weights = w.reshape(self.weights.shape) if fold else w


class DropOut(DenseLayer):
    def __init__(self, brain, inputs, neurons, dropout, position, activation):
        DenseLayer.__init__(self, brain, inputs, neurons, position, activation)

        self.dropchance = 1 - dropout
        self.mask = None

    def feedforward(self, questions):
        self.inputs = rtm(questions)
        self.mask = np.random.uniform(0, 1, self.biases.shape) < self.dropchance
        Z = (np.dot(self.inputs, self.weights) + self.biases) * self.mask
        self.output = self.activation(Z)
        return self.output

    def predict(self, question):
        return DenseLayer.predict(self, question) * self.dropchance

    def backpropagation(self):

        if self.brain.mu:
            self.velocity += self.gradients
        self.gradients = (np.dot(self.inputs.T, self.error) / self.brain.m) * self.mask

        return np.dot(self.error, self.weights.T * self.mask.T)


class InputLayer(_LayerBase):

    def __init__(self, brain, inshape):
        _LayerBase.__init__(self, brain, position=0, activation="linear")
        self.neurons = inshape
        self.trainable = False

    def feedforward(self, questions):
        """
        Passes the unmodified input matrix

        :param questions: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = self.output = questions
        return questions

    def predict(self, stimuli):
        """
        Passes the unmodified input matrix.

        This method has no side-effects. Used in prediction and testing.

        :param stimuli: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = stimuli
        return stimuli

    def backpropagation(self): pass

    def weight_update(self): pass

    def receive_error(self, error_vector: np.ndarray): pass

    def shuffle(self): pass

    def get_weights(self, unfold=True):
        return None

    def set_weights(self, w, fold=True):
        pass

    @property
    def outshape(self):
        return self.neurons

    @property
    def weights(self):
        warnings.warn("Queried weights of an InputLayer!", RuntimeWarning)
        return None


class Recurrent(_FFLayer):

    def __init__(self, brain, inputs, neurons, position, activation, return_seq):
        _FFLayer.__init__(self, brain, neurons, position, activation)
        self.Z = inputs + neurons
        self.cache = self.Cache(inputs, neurons)
        self.time = 0
        self.return_seq = return_seq

        self.weights = None
        self.biases = None
        self.gradients = None
        self.nabla_b = None
        self.velocity = None

    def receive_error(self, error_matrix: np.ndarray):
        if self.return_seq:
            self.error = error_matrix.transpose(1, 0, 2)
        else:
            self.error = np.zeros((self.time, self.brain.m, self.neurons), dtype=floatX)
            self.error[-1] += error_matrix

    def weight_update(self) -> None:
        eta = self.brain.eta / self.brain.m

        if self.brain.mu:
            self.velocity *= self.brain.mu
            self.velocity += self.gradients * eta
            self.weights -= self.velocity
            # self.biases -= eta * self.nabla_b
        else:
            self.weights -= eta * self.gradients
            # self.biases -= eta * self.nabla_b

    @abc.abstractmethod
    def feedforward(self, stimuli: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, stimuli: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def backpropagation(self) -> np.ndarray:
        raise NotImplementedError

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
            self.innards = np.zeros((self.k, time, self.m, self.n), dtype=floatX)
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
            self[key] = np.zeros((self.m, self.n), dtype=floatX)


class LSTM(Recurrent):

    def __init__(self, brain, neurons, inputs, position, return_seq):
        Recurrent.__init__(self, brain, inputs, neurons, position, return_seq=return_seq, activation="tanh")

        self.G = 3 * neurons  # size of 3 gates -> needed for slicing

        self.weights = white(self.Z, neurons * 4)
        self.biases = white(1, neurons * 4)

        self.gate_W_gradients = np.zeros_like(self.weights)
        self.gate_b_gradients = np.zeros_like(self.biases)

        self.fanin = inputs

    def feedforward(self, stimuli: np.ndarray):

        def timestep(Z, C):
            preact = Z.dot(self.weights) + self.biases
            f, i, o = np.split(sigmoid(preact[:self.G]), 3)
            cand = tanh(preact[self.G:])
            C = C * f + i * cand
            thC = tanh(C)
            h = thC * o
            return h, C, (thC, f, i, o, cand)

        # transposition is neccesary because the first datadim is not time,
        # but the batch index. (compatibility with CsxData and Keras)
        stimuli = np.transpose(stimuli, (1, 0, 2))
        self.time = self.inputs.shape[0]
        self.cache.reset(time=self.time)
        output = np.zeros(self.outshape)
        state = np.zeros(self.outshape)
        for time in range(self.time):
            concatenated_inputs = np.concatenate(stimuli[time], output)
            output, state, cache = timestep(concatenated_inputs, state)

            self.cache["outputs"][time] = output
            self.cache["states"][time] = state
            self.cache["tanh states"][time] = cache[0]
            self.cache["gate forget"][time] = cache[1]
            self.cache["gate input"][time] = cache[2]
            self.cache["gate output"][time] = cache[3]
            self.cache["candidates"][time] = cache[4]
            self.cache["Z"][time] = concatenated_inputs

        # I'm not sure about the output,
        # because either the whole output sequence can be returned -> this keeps the input dims
        # or just the last output -> this leads to dim reduction
        if self.return_seq:
            self.output = self.cache["outputs"].transpose(1, 0, 2)
            return self.output
        else:
            self.output = self.cache["outputs"][-1]
            return self.output

    def predict(self, stimuli: np.ndarray):

        def timestep(z, C):
            """
            One step in time

            Where
            f, i, o are the forget gate, input gate and output gate activations.
            f determines how much information we forget from the previous cell state (0.0 means forget all)
            i determines how much information we add to the current cell state from the z stack.
            o determines how much information we provide as layer output from the cell state.
            candidate is a cell state candidate, it is added to cell state after appliing i to it.
            :param z: the stack: current input concatenated with the output from the previous timestep
            :param C: the cell state from the previous timestep
            """
            # Calculate the gate activations
            preact = z.dot(self.weights)
            # TODO: rewrite with np.split!!!
            f, i, o = sigmoid(preact[:self.G]).reshape(self.Z, 3, self.neurons).transpose(1, 0, 2)
            # Calculate the cell state candidate
            candidate = tanh(preact[self.G:])
            # Apply forget gate to the previus cell state receives as a parameter
            # Apply input gate to cell state candidate, then update cell state
            C = C * f + i * candidate
            # Apply output gate to tanh of cell state. This is the layer output at timestep <t>
            return o * tanh(C)

        # Initialize cell state and layer outputs to all 0s
        state = np.zeros(self.outshape)
        outputs = np.zeros((self.time, self.brain.m, self.neurons))
        # Transposition is needed so timestep becomes dim0, 4 compatibility with keras and csxdata
        stimuli = np.transpose(stimuli, (1, 0, 2))
        for time, inputs in enumerate(stimuli):
            stack = np.column_stack((inputs, outputs[-1]))
            outputs[time], state = timestep(stack, state)

        if self.return_seq:
            return outputs.transpose(1, 0, 2)
        else:
            return outputs[-1]

    def backpropagation(self):

        def bptt_timestep(t, dy, dC):
            assert dC is 0 and t == self.time
            cch = self.cache
            dC = tanh.derivative(cch["states"][t]) * cch["gate output"][t] * dy + dC
            do = sigmoid.derivative(cch["gate output"][t]) * cch["tanh states"] * dC
            di = sigmoid.derivative(cch["gate input"][t]) * cch["candidates"] * dC
            df = sigmoid.derivative(cch["gate forget"][t]) * cch["states"][t-1] * dC
            dcand = tanh.derivative(cch["cadidates"][t]) * cch["gate input"][t] * dC
            deltas = np.concatenate((df, di, df, do, dcand), axis=-1)
            dZ = deltas.dot(self.weights.T)
            gW = cch["Z"][t].T.dot(deltas)
            return gW, dZ, dC

        self.gate_W_gradients = np.zeros_like(self.weights)
        dstate = 0  # so bptt dC receives + 0 @ time == self.time
        deltaY = np.zeros((self.brain.m, self.neurons), dtype=floatX)
        deltaX = np.zeros((self.time, self.brain.m, self.inputs), dtype=floatX)

        for time in range(self.time, -1, -1):
            if time < self.time:
                dstate *= self.cache["gate forget"][time+1]
            deltaY += self.error[time]
            gradW, deltaZ, dstate = bptt_timestep(time, deltaY, dstate)
            deltaY = deltaZ[self.neurons:]
            deltaX[time] = deltaZ[self.neurons:]
            self.gate_W_gradients += gradW

        return deltaX


class RLayer(Recurrent):

    def __init__(self, brain, inputs, neurons, position, activation, return_seq=False):
        Recurrent.__init__(self, brain, inputs, neurons, position, activation, return_seq=return_seq)

        self.weights = white(self.Z, self.neurons)
        self.biases = np.zeros((self.neurons,), dtype=floatX)
        self.gradients = np.zeros_like(self.weights)
        self.nabla_b = np.zeros_like(self.biases)
        self.velocity = np.zeros_like(self.weights)

    def feedforward(self, questions: np.ndarray):
        self.inputs = questions.transpose(1, 0, 2)
        self.time = self.inputs.shape[0]
        self.cache.reset(batch_size=self.brain.m, time=self.time)

        def timestep(Z):
            wsum = Z.dot(self.weights) + self.biases
            return self.activation(wsum)

        for t in range(self.time):
            self.cache["Z"][t] = np.concatenate((self.inputs[t], self.cache["outputs"][t-1]), axis=1)
            self.cache["outputs"][t] = timestep(self.cache["Z"][t])

        if self.return_seq:
            self.output = self.cache["outputs"].transpose(1, 0, 2)
        else:
            self.output = self.cache["outputs"][-1]

        return self.output

    def predict(self, stimuli: np.ndarray) -> np.ndarray:
        stimuli = stimuli.transpose(1, 0, 2)
        time, m = stimuli.shape[:2]

        def timestep(x, h):
            Z = np.concatenate((x, h), axis=1)
            h = self.activation(Z.dot(self.weights) + self.biases)
            return h

        outputs = np.zeros((time, m, self.neurons))
        for t in range(time):
            outputs[t] = timestep(stimuli[t], outputs[t-1])

        if self.return_seq:
            return outputs.transpose(1, 0, 2)
        else:
            return outputs[-1]

    def backpropagation(self):
        """Backpropagation through time (BPTT)"""

        def bptt_timestep(t, dY, dh):
            """
            :param t: the timestep indicator
            :param dY: dC/dY_t -> gradient of the layer output (if any)
            :param dh: dC/dY_t+1 -> state gradient flowing backwards in time

            :return: gR: dC/dR @ timestep <t>; dX dC/dX_{t}; dh: state gradient flowing backwards
            """
            delta_now = (dY + dh) * self.activation.derivative(self.cache["outputs"][t])
            gR = self.cache["Z"][t].T.dot(delta_now)
            dZ = delta_now.dot(self.weights.T)
            dX = dZ[:, :-self.neurons]
            dh = dZ[:, -self.neurons:]
            return gR, dX, dh

        # gradient of the cost wrt the weights: dC/dW
        self.gradients = np.zeros_like(self.weights)
        # gradient of the cost wrt to biases: dC/db
        self.nabla_b = self.error[-1].sum(axis=0)
        # the gradient flowing backwards in time
        error = np.zeros_like(self.error[-1])
        # the gradient wrt the whole input tensor: dC/dX = dC/dY_{l-1}
        delta_X = np.zeros_like(self.inputs)

        for time in range(self.time-1, -1, -1):
            grad_R, delta_X[time], error = bptt_timestep(time, self.error[time], error)
            self.gradients += grad_R
            # self.nabla_b += gradient.sum(axis=0)

        return delta_X.transpose(1, 0, 2)


class EchoLayer(RLayer):
    def __init__(self, brain, inputs, neurons, position, activation, return_seq=False,
                 p=0.1):
        RLayer.__init__(self, brain, inputs, neurons, position, activation, return_seq)
        self.weights = np.random.binomial(1., p, size=self.weights.shape).astype(floatX)
        self.weights *= np.random.randn(*self.weights.shape)  # + 1.)
        self.trainable = False

    def weight_update(self):
        pass

    def get_weights(self, unfold=True):
        return np.array([[]])

    def set_weights(self, w, fold=True):
        pass

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

        def backpropagation(self):
            """
            Calculates the error of the previous layer.
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

        def backpropagation(self):
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

    class AboLayer(_LayerBase):
        def __init__(self, brain, position, activation):
            _LayerBase.__init__(self, brain, position, activation)
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

        def backpropagation(self) -> np.ndarray:
            pass

        def weight_update(self) -> None:
            pass

        def predict(self, stimuli: np.ndarray) -> np.ndarray:
            pass

        def outshape(self):
            return ...
