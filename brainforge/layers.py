import abc

import numpy as np
from scipy.ndimage import convolve

from ..util import activations
from ..util import sigmoid, tanh
from ..util import l1term, l2term, outshape, calcsteps, white

from csxdata import floatX
from csxdata.utilities.nputils import maxpool, ravel_to_matrix as rtm


class _LayerBase(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, brain, position, activation):
        self.brain = brain
        self.position = position
        self.inputs = None
        if isinstance(activation, str):
            self.activation = activations[activation]
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


class _FFLayer(_LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, brain, neurons: int, position: int, activation):
        _LayerBase.__init__(self, brain, position, activation)

        self.neurons = neurons
        self.inputs = None

        self.output = np.zeros((neurons,))
        self.error = np.zeros((neurons,))

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

        self.weights = white(inputs, neurons)
        self.gradients = np.zeros_like(self.weights)
        self.velocity = np.zeros_like(self.weights)
        self.biases = np.zeros((1, neurons), dtype=float)
        self.N = 0  # current batch size
        # print("<FF", self.activation, "layer> created with input size {} and output size {} @ position {}"
        #       .format(inputs, neurons, position))

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
            self.velocity += self.gradients
        # (dC / dW) = error * (dZ / dW), where error = (dMSE / dA) * (dA / dZ)
        self.gradients = np.dot(self.inputs.T, self.error) / self.brain.m
        # (dC / dW) = error * (dZ / dA_), where A_ is the previous output
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
            np.subtract(self.weights, self.brain.mu * (self.velocity + self.gradients) * self.brain.eta,
                        out=self.weights)

        def descend_on_gradient():
            np.subtract(self.weights, self.gradients * self.brain.eta,
                        out=self.weights)

        def modify_biases():
            np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                        out=self.biases)

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

    def shuffle(self):
        ws = self.weights.shape
        self.weights = np.random.randn(*ws) / np.sqrt(ws[0])


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

    def feedforward(self, questions):
        """
        Passes the unmodified input matrix

        :param questions: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = questions
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

    @property
    def outshape(self):
        return self.neurons


class Experimental:

    class LSTM(_FFLayer):
        def __init__(self, brain, neurons, inputs, position, activation="tanh"):
            _FFLayer.__init__(self, brain, neurons, position, activation)

            print("Warning! CsxNet LSTM Layer is experimental!")

            self.Z = neurons + inputs  # size of column stacked <inputs, outputs>
            self.G = 3 * neurons  # size of 3 gates -> needed for slicing

            self.weights = white(self.Z, neurons * 4)
            self.biases = white(1, neurons * 4)

            self.gate_W_gradients = np.zeros_like(self.weights)
            self.gate_b_gradients = np.zeros_like(self.biases)

            self.output = None
            self.cache = self.Cache(self.brain.m, self.Z)

            self.time = 0
            self.fanin = inputs

        def feedforward(self, stimuli: np.ndarray):
            # transposition is neccesary because the first datadim is not time,
            # but the batch index. (compatibility with CsxData and Keras)
            self.inputs = np.transpose(stimuli, (1, 0, 2))
            self.time = self.inputs.shape[0]
            self.cache.reset(time=self.time)
            # TODO: create a pretty nested function
            for time in range(self.time):
                stimulus = stimuli[time]
                state_yesterday = self.cache["states"][time-1]

                X = np.column_stack((stimulus, self.cache["outputs"][-1]))

                gates = X.dot(self.weights) + self.biases
                # gates: forget, input, output
                # TODO: rewrite this with np.split and omit transposition
                gf, gi, go = np.transpose(sigmoid(gates[:, self.neurons * 3])
                                          .reshape(self.fanin, 3, self.neurons),
                                          axes=(1, 0, 2))
                # state candidate
                candidate = gates[:, 3 * self.neurons:] = tanh(gates[:, 3 * self.neurons:])

                state = gf * state_yesterday + gi * candidate
                tanh_state = tanh(state)
                output = go * tanh_state

                self.cache["outputs"][time] += output
                self.cache["states"][time] += state
                self.cache["tanh states"][time] += tanh_state
                self.cache["gate forget"][time] += gf
                self.cache["gate input"][time] += gf
                self.cache["gate output"][time] += gf

            # I'm not sure about the output,
            # because either the whole output sequence can be returned -> this keeps the input dims
            # or just the last output -> this leads to dim reduction
            if 1:
                self.output = self.cache["outputs"][-1]
            else:
                self.output = self.cache["outputs"]

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
                f, i, o = sigmoid(preact[:self.G]).reshape(self.Z, 3, self.neurons).transpose(1, 0, 2)
                # Calculate the cell state candidate
                candidate = tanh(preact[self.G:])
                # Apply forget gate to the previus cell state receives as a parameter
                C *= f
                # Apply input gate to cell state candidate, then update cell state
                C += i * candidate
                # Apply output gate to tanh of cell state. This is the layer output at timestep <t>
                return o * tanh(C)

            # Initialize cell state and layer outputs to all 0s
            state = np.zeros(self.outshape)
            outputs = np.zeros(self.outshape)
            # Transposition is needed so timestep becomes dim0, 4 compatibility with keras and csxdata
            stimuli = np.transpose(stimuli, (1, 0, 2))
            for inputs in stimuli:
                stack = np.column_stack((inputs, outputs))
                outputs, state = timestep(stack, state)
            return outputs

        def backpropagation(self):

            def bptimestep(t, dy, dC):
                assert dC is 0 and t == self.time
                cch = self.cache
                dC = tanh.derivative(cch["states"][t]) * cch["gate output"][t] * dy + dC
                do = sigmoid.derivative(cch["gate output"][t]) * cch["tanh states"] * dC
                di = sigmoid.derivative(cch["gate input"][t]) * cch["candidates"] * dC
                df = sigmoid.derivative(cch["gate forget"][t]) * cch["states"][t-1] * dC
                dcand = tanh.derivative(cch["cadidates"][t]) * cch["gate input"][t] * dC
                deltas = np.concatenate((df, di, df, do, dcand), axis=-1)
                dZ = deltas.dot(self.weights.T)
                gW = self.inputs[t].T.dot(deltas)
                return gW, dZ, dC

            self.gate_W_gradients = np.zeros_like(self.weights)
            dstate = 0  # so bptt dC receives + 0 @ time == self.time
            deltaY = np.zeros(self.outshape, dtype=floatX)
            deltaX = np.zeros((self.time, self.brain.m, self.inputs), dtype=floatX)

            for time in range(self.time, -1, -1):
                if time < self.time:
                    dstate *= self.cache["gate forget"][time+1]
                gradW, deltaZ, dstate = bptimestep(time, deltaY, dstate)
                deltaY = deltaZ[self.neurons:]
                deltaX[time] = deltaZ[self.neurons:]
                self.gate_W_gradients += gradW

            if 1:
                sm = deltaX.sum(axis=0)
                assert sm.shape == (self.brain.m, self.inputs) == self.brain.layers[self.position-1].shape
                return sm
            else:
                return deltaX

        def weight_update(self):
            # Update weights and biases
            np.subtract(self.weights, self.gate_W_gradients * self.brain.eta,
                        out=self.weights)
            np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                        out=self.biases)

        def receive_error(self, error_vector: np.ndarray):
            time, neurons = self.outshape
            self.error = error_vector.reshape(self.brain.m, time, neurons).transpose(1, 0, 2)
            self.error *= self.activation.derivative(self.output)

        def shuffle(self):
            pass

        @property
        def outshape(self):
            return self.time, self.neurons

        class Cache:
            def __init__(self, m, Z):
                self.keys = ("outputs", "tanh states", "gate output", "states",
                             "candidates", "gate input", "gate forget")
                self.k = len(self.keys)
                self.m = m
                self.Z = Z
                self.t = None
                self.innards = None

            def assertkey(self, item):
                if item not in self.keys:
                    raise IndexError("There is no cache named {}".format(item))

            def reset(self, time=None):
                if self.t is None and time is None:
                    raise RuntimeError("No previous time information is available! Supply time parameter!")
                if time is None:
                    time = self.t
                self.t = time
                self.innards = np.zeros((self.k, time, self.m, self.Z), dtype=floatX)

            def __getitem__(self, item):
                self.assertkey(item)
                return self.innards[self.keys.index(item)]

            def __setitem__(self, key, value):
                self.assertkey(key)
                if value.shape != (self.m, self.Z):
                    raise ValueError("Shapes differ: self: {} != {} :value"
                                     .format(self.oneshape, value.shape))
                self.innards[self.keys.index(key)] = value

            def __delitem__(self, key):
                self.assertkey(key)
                self[key] = np.zeros((self.m, self.Z), dtype=floatX)

    class RLayer(DenseLayer):
        def __init__(self, brain, inputs, neurons, time_truncate, position, activation):
            DenseLayer.__init__(self, brain, inputs, neurons, position, activation)

            self.time_truncate = time_truncate
            self.rweights = np.random.randn(neurons, neurons)
            self.rgradients = np.zeros_like(self.rweights)

        def feedforward(self, questions):
            self.inputs = rtm(questions)
            time = questions.shape[0]
            self.output = np.zeros((time + 1, self.outshape))
            preact = np.dot(self.inputs, self.weights)
            for t in range(time):
                self.output[t] = self.activation(
                    preact[t] + np.dot(self.output[t - 1], self.rweights)
                )
            return self.output

        def backpropagation(self):
            """Backpropagation through time (BPTT)"""
            T = self.error.shape[0]
            self.gradients = np.zeros(self.weights.shape)
            self.rgradients = np.zeros(self.rweights.shape)
            prev_error = np.zeros_like(self.inputs)
            for t in range(0, T, step=-1):
                t_delta = self.error[t]
                for bptt in range(max(0, t - self.time_truncate), t + 1, step=-1):
                    # TODO: check the order of parameters. Transposition possibly needed somewhere
                    self.rgradients += np.outer(t_delta, self.output[bptt - 1])
                    self.gradients += np.dot(self.gradients, self.inputs) + t_delta
                    t_delta = self.rweights.dot(t_delta) * self.activation.derivative(self.output[bptt - 1])
                prev_error[t] = t_delta
            return prev_error

        def receive_error(self, error):
            """
            Transforms the received error tensor to adequate shape and stores it.

            :param error: T x N shaped, where T is time and N is the number of neurons
            :return: None
            """
            self.error = rtm(error)

        def weight_update(self):
            self.weights -= self.brain.eta * self.gradients
            self.rweights -= self.brain.eta * self.rgradients

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
