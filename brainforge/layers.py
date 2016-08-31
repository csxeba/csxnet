import abc

import numpy as np
from scipy.ndimage import convolve

from ._activations import activation as actfns, sigmoid, tanh
from ..util import l1term, l2term, outshape, calcsteps, white

from csxdata.utilities.nputils import maxpool, ravel_to_matrix as rtm
from csxdata.const import floatX


class _LayerBase(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, brain, position, activation):
        self.brain = brain
        self.position = position
        if isinstance(activation, str):
            self.activation = actfns[activation]
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


class _FFLayer(_LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, brain, neurons: int, position: int, activation):
        _LayerBase.__init__(self, brain, position, activation)

        self.neurons = neurons
        self.outshape = (neurons,)
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


class DenseLayer(_FFLayer):
    """Just your regular Densely Connected Layer

    Aka Dense (Keras), Fully Connected, Feedforward, etc.
    Elementary building stone for the Multilayer Perceptron model.
    """
    def __init__(self, brain, inputs, neurons, position, activation):
        _FFLayer.__init__(self, brain=brain,
                          neurons=neurons, position=position,
                          activation=activation)

        self.weights = white(inputs, neurons)
        self.gradients = np.zeros_like(self.weights)
        self.velocity = np.zeros_like(self.weights)
        self.biases = np.zeros((1, neurons), dtype=float)
        self.inputs = None
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


class InputLayer(_FFLayer):
    def __init__(self, brain, inshape):
        _FFLayer.__init__(self, brain=brain, neurons=0, position=1, activation="linear")
        self.outshape = inshape

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
        return stimuli

    def backpropagation(self): pass

    def weight_update(self): pass

    def receive_error(self, error_vector: np.ndarray): pass

    def shuffle(self): pass


class Experimental:

    class LSTM(_FFLayer):
        def __init__(self, brain, neurons, inputs, position, activation="tanh"):
            _FFLayer.__init__(self, brain, neurons, position, activation)

            print("Warning! CsxNet LSTM Layer is experimental!")

            self.Z = neurons + inputs

            self.weights = white(self.Z, neurons * 4)
            self.biases = white(1, neurons * 4)

            self.gate_W_gradients = np.zeros_like(self.weights)
            self.gate_b_gradients = np.zeros_like(self.biases)

            self.output = None
            self.cache = None

            self.time = 0
            self.fanin = inputs

        def feedforward(self, stimuli: np.ndarray):
            self.inputs = stimuli
            self.time = stimuli.shape[1]
            self.cache = {"outputs": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX),
                          "states": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX),
                          "candidates": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX),
                          "gate forget": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX),
                          "gate input": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX),
                          "gate output": np.zeros((self.time, self.brain.m, self.Z), dtype=floatX)}
            # this step might be neccesary if the first datadim is not time, but the batch index
            stimuli = np.transpose(stimuli, (1, 0, 2))

            for time in range(self.time):
                stimulus = stimuli[time]
                state_yesterday = self.cache["states"][time-1]

                X = np.column_stack((stimulus, self.cache["outputs"][-1]))

                gates = X.dot(self.weights) + self.biases
                # gates: forget, input, output
                gates[:, :self.neurons * 3] = sigmoid(gates[:, self.neurons * 3])
                # state candidate
                gates[:, 3 * self.neurons:] = tanh(gates[:, 3 * self.neurons:])

                # This is basically a slicing step
                gf, gi, go, candidate = np.transpose(gates.reshape(self.fanin, 4, self.neurons), axes=(1, 0, 2))
                state = gf * state_yesterday + gi * candidate
                tanh_state = tanh(state)
                output = go * tanh_state

                self.cache["outputs"][time] += output
                self.cache["states"][time] += state
                self.cache["tanh_states"][time] += tanh_state
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
            pass

        def weight_update(self):
            # Update weights and biases
            np.subtract(self.weights, self.gate_W_gradients * self.brain.eta,
                        out=self.weights)
            np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                        out=self.biases)

        def backpropagation(self):
            error_now = np.zeros_like(self.error)
            gate_errors = []
            backpass = []
            dstate = np.zeros_like(self.cache["states"][0])

            # No momentum (yet)
            self.gate_W_gradients = np.zeros_like(self.weights)
            self.gate_b_gradients = np.zeros_like(self.biases)

            # backprop through time
            for t in range(0, self.time, -1):
                gf = self.cache["gate forget"][t]
                gi = self.cache["gate input"][t]
                cand = self.cache["candidates"][t]
                go = self.cache["gate output"][t]
                error_now += self.error[t] + error_now

                dgo = sigmoid.derivative(go) * self.cache["tanh_states"][t] * error_now
                dstate = tanh.derivative(cand) * (go * error_now + dstate)
                dgf = sigmoid.derivative(gf) * (self.cache["states"][t - 1] * dstate)
                dgi = sigmoid.derivative(gi) * (cand * dstate)
                dcand = tanh.derivative(cand) * (gi * dstate)

                gate_errors.append(np.column_stack((dgf, dgi, dgo, dcand)))
                self.gate_W_gradients += self.inputs.T.dot(gate_errors[-1])
                self.gate_b_gradients += gate_errors[-1]
                # Folding the (fanin, 4*neurons) part into (4, neurons)
                # then summing the 4 matrices into 1 and getting (fanin, neurons)
                dZ = gate_errors[-1].dot(self.weights.T)

                backpass.append(dZ[:, :self.neurons])  # error wrt to the previous input
                error_now = dZ[:, self.neurons:]  # error wrt to own output tomorrow
                dstate = gf * dstate

            prev_error = np.dot(gate_errors, self.weights.T)
            return prev_error

        def receive_error(self, error_vector: np.ndarray):
            self.error = error_vector.reshape(self.brain.m, *self.outshape) * \
                         self.activation.derivative(self.output)

        def shuffle(self):
            pass

    class RLayer(DenseLayer):
        def __init__(self, brain, inputs, neurons, time_truncate, position, activation):
            DenseLayer.__init__(self, brain, inputs, neurons, position, activation)

            self.time_truncate = time_truncate
            self.rweights = np.random.randn(neurons, neurons)
            self._grad_rweights = np.zeros_like(self.rweights)

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
            self._grad_weights = np.zeros(self.weights.shape)
            self._grad_rweights = np.zeros(self.rweights.shape)
            prev_error = np.zeros_like(self.inputs)
            for t in range(0, T, step=-1):
                t_delta = self.error[t]
                for bptt in range(max(0, t - self.time_truncate), t + 1, step=-1):
                    # TODO: check the order of parameters. Transposition possibly needed somewhere
                    self._grad_rweights += np.outer(t_delta, self.output[bptt - 1])
                    self._grad_weights += np.dot(self._grad_weights, self.inputs) + t_delta
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
            self.weights -= self.brain.eta * self._grad_weights
            self.rweights -= self.brain.eta * self._grad_rweights

    class PoolLayer(_VecLayer):
        def __init__(self, brain, inshape, fshape, stride, position):
            _VecLayer.__init__(self, brain=brain,
                               inshape=inshape, fshape=fshape,
                               stride=stride, position=position,
                               activation="linear")
            self.outshape = (self.inshape[0], self.outshape[0], self.outshape[1])
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

    class ConvLayer(_VecLayer):
        def __init__(self, brain, fshape, inshape, num_filters, stride, position, activation):
            _VecLayer.__init__(self, brain=brain,
                               inshape=inshape, fshape=fshape,
                               stride=stride, position=position,
                               activation=activation)

            chain = """TODO: fix convolution. Figure out backprop. Unify backprop and weight update. (?)"""
            print(chain)
            self.inputs = np.zeros(self.inshape)
            self.outshape = num_filters, self.outshape[0], self.outshape[1]
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
