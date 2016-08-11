import abc

import numpy as np
from scipy.ndimage import convolve

from .activations import Linear

from ..utilities.nputils import maxpool, ravel_to_matrix as rtm, outshape, calcsteps
from ..utilities.pure import l1term, l2term


class _LayerBase(abc.ABC):
    """Abstract base class for all layer type classes"""
    def __init__(self, brain, position, activation):
        self.brain = brain
        self.position = position
        self.activation = activation()

    def feedforward(self, stimuli: np.ndarray) -> np.ndarray: pass

    def predict(self, stimuli: np.ndarray) -> np.ndarray: pass

    def backpropagation(self) -> np.ndarray: pass

    def weight_update(self) -> None: pass

    def receive_error(self, error_vector: np.ndarray) -> None: pass

    def shuffle(self) -> None: pass


class _VecLayer(_LayerBase):
    """Base class for layer types, which operate on tensors
     and are sparsely connected"""
    def __init__(self, brain, inshape: tuple, fshape: tuple,
                 stride: int, position: int,
                 activation: type):
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


class _FCLayer(_LayerBase):
    """Base class for the fully connected layer types"""
    def __init__(self, brain, neurons: int, position: int, activation: type):
        _LayerBase.__init__(self, brain, position, activation)

        self.neurons = neurons
        self.outshape = (neurons,)
        self.inputs = None

        self.output = np.zeros((neurons,))
        self.error = np.zeros((neurons,))


class PoolLayer(_VecLayer):
    def __init__(self, brain, inshape, fshape, stride, position):
        _VecLayer.__init__(self, brain=brain,
                           inshape=inshape, fshape=fshape,
                           stride=stride, position=position,
                           activation=Linear)
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
        result = np.zeros((m*f*len(self.coords),))
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
        result = np.zeros((m*f*len(self.coords),))
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

        prev = self.brain.layers[self.position-1]
        return deltas * prev.activation.derivative(prev.output)

    def receive_error(self, error_matrix):
        """
        Folds a received error matrix.
        :param error_matrix: backpropagated errors from the next layer
        :return: None
        """
        self.error = error_matrix.reshape([self.brain.m] + [self.outshape[0]] +
                                          [np.prod(self.outshape[1:])])


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
        self.filters = np.random.randn(num_filters, np.prod(fshape)) / np.sqrt(np.prod(inshape))
        self.grad_filters = np.zeros_like(self.filters)
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
        prev = self.brain.layers[self.position-1]
        return deltas * prev.activation.derivative(prev.output)

    def receive_error(self, error_matrix: np.ndarray):
        """
        Fold the received error matrix.

        :param error_matrix: numpy.ndarray: backpropagated errors
        :return: None
        """
        self.error = error_matrix.reshape([self.brain.m] + list(self.outshape))

    def weight_update(self):
        """
        Updates convolutional filter weights with the calculated _grad_weights.

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


class FFLayer(_FCLayer):
    def __init__(self, brain, inputs, neurons, position, activation):
        _FCLayer.__init__(self, brain=brain,
                          neurons=neurons, position=position,
                          activation=activation)

        self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs)
        self._grad_weights = np.zeros_like(self.weights)
        self._old_grads = np.zeros_like(self.weights)
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
        self._old_grads = np.copy(self._grad_weights)
        self._grad_weights = np.dot(self.inputs.T, self.error) / self.brain.m

        prev = self.brain.layers[self.position-1]
        posh = [self.error.shape[0]] + list(prev.outshape)
        prev_error = np.dot(self.error, self.weights.T).reshape(posh)
        prev_error *= prev.activation.derivative(prev.output)

        return prev_error

    def weight_update(self):
        """
        Performs Stochastic Gradient Descent by subtracting a portion of the
        calculated gradients from the weights and biases.

        :return: None
        """
        # Apply L2 regularization, aka weight decay
        l1 = l1term(self.brain.eta, self.brain.lmbd1, self.brain.N)
        l2 = l2term(self.brain.eta, self.brain.lmbd2, self.brain.N)
        if self.brain.lmbd2:
            self.weights *= l2
        if self.brain.lmbd1:
            self.weights -= l1 * np.sign(self.weights)

        # Update weights and biases
        np.subtract(self.weights,
                    (self._grad_weights + self.brain.mu * self._old_grads) *
                    self.brain.eta,
                    out=self.weights)
        np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                    out=self.biases)

    def receive_error(self, error):
        """
        Saves the received error matrix.

        The received matrix should not be folded, since FFLayer should only be
        followed by FFLayer.

        :param error: numpy.ndarray: 2D matrix of errors
        :return: None
        """
        self.error = error

    def shuffle(self):
        ws = self.weights.shape
        self.weights = np.random.randn(*ws) / np.sqrt(ws[0])


class DropOut(FFLayer):
    def __init__(self, brain, inputs, neurons, dropout, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)

        self.dropchance = 1 - dropout
        self.mask = None

    def feedforward(self, questions):
        self.inputs = rtm(questions)
        self.mask = np.random.uniform(0, 1, self.biases.shape) < self.dropchance
        Z = (np.dot(self.inputs, self.weights) + self.biases) * self.mask
        self.output = self.activation(Z)
        return self.output

    def predict(self, question):
        return FFLayer.predict(self, question) * self.dropchance

    def backpropagation(self):

        self._old_grads = np.copy(self._grad_weights)
        self._grad_weights = (np.dot(self.inputs.T, self.error) / self.brain.m) * self.mask

        prev = self.brain.layers[self.position-1]
        posh = [self.error.shape[0]] + list(prev.outshape)
        prev_error = np.dot(self.error, self.weights.T * self.mask.T).reshape(posh)
        prev_error *= prev.activation.derivative(prev.output)

        return prev_error

    def weight_update(self):
        # Apply L1/L2 regularization, aka weight decay
        l1 = l1term(self.brain.eta, self.brain.lmbd1, self.brain.N)
        l2 = l2term(self.brain.eta, self.brain.lmbd2, self.brain.N)
        if self.brain.lmbd2:
            self.weights *= l2
        if self.brain.lmbd1:
            self.weights -= l1 * np.sign(self.weights)

        np.subtract(self.weights,
                    (self._grad_weights + self.brain.mu * self._old_grads) *
                    self.brain.eta,
                    out=self.weights)
        np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                    out=self.biases)


class RLayer(FFLayer):
    def __init__(self, brain, inputs, neurons, time_truncate, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)

        self.time_truncate = time_truncate
        self.rweights = np.random.randn(neurons, neurons)
        self._grad_rweights = np.zeros_like(self.rweights)

    def feedforward(self, questions):
        self.inputs = rtm(questions)
        time = questions.shape[0]
        self.output = np.zeros((time+1, self.outshape))
        preact = np.dot(self.inputs, self.weights)
        for t in range(time):
            self.output[t] = self.activation(
                preact[t] + np.dot(self.output[t-1], self.rweights)
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
            for bptt in range(max(0, t-self.time_truncate), t+1, step=-1):
                # TODO: check the order of parameters. Transposition possibly needed somewhere
                self._grad_rweights += np.outer(t_delta, self.output[bptt - 1])
                self._grad_weights += np.dot(self._grad_weights, self.inputs) + t_delta
                t_delta = self.rweights.dot(t_delta) * self.activation.derivative(self.output[bptt-1])
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

    # noinspection PyUnresolvedReferences
    def __bptt_reference(self, x, y):
        """FROM www.wildml.com"""
        T = len(y)
        # Perform forward propagation
        # Catch network output and hidden output
        o, s = self.forward_propagation(x)
        # We accumulate the _grad_weights in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        # OMG THIS SIMPLY CALCULATES (output - target*)...
        # Which by the way is the grad of Xent wrt to the outweights
        # aka the output error
        # *provided that target is not converted to 1-hot vectors
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.

        for t in np.arange(T)[::-1]:
            # This is the sum of the output weights' _grad_weights
            dLdV += np.outer(delta_o[t], s[t].T)
            # backpropagating to the hiddens (Weights * Er) * tanh'(Z)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.time_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    # noinspection PyUnresolvedReferences
    def __forward_propagation(self, x):
        """FROM www.wildml.com"""
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outsize at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            # aka self.U.dot(x)
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]


class InputLayer(_FCLayer):
    def __init__(self, brain, inshape):
        _FCLayer.__init__(self, brain=brain, neurons=0, position=1, activation=Linear)
        self.outshape = inshape

    def feedforward(self, questions):
        """
        Passes the unmodified input matrix

        :param questions: numpy.ndarray
        :return: numpy.ndarray
        """
        self.inputs = questions
        return questions

    def mp_feedforward(self, questions):
        return self.feedforward(questions)

    def predict(self, stimuli):
        """
        Passes the unmodified input matrix.

        This method has no side-effects. Used in prediction and testing.

        :param stimuli: numpy.ndarray
        :return: numpy.ndarray
        """
        return stimuli
