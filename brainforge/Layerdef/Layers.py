import multiprocessing as mp

import numpy as np
from scipy.ndimage import convolve

from ..Layerdef._LayerBase import _VecLayer, _FCLayer
from ..Utility.activations import Linear
from ..Utility.operations import maxpool
from ..Utility.utility import ravel_to_matrix as ravtm, l2term


class PoolLayer(_VecLayer):
    def __init__(self, brain, inshape, fshape, stride, position):
        _VecLayer.__init__(self, brain=brain,
                           inshape=inshape, fshape=fshape,
                           stride=stride, position=position,
                           activation=Linear)
        self.outshape = (self.inshape[0], self.outshape[0], self.outshape[1])
        self.backpass_filter = None
        print("<PoolLayer> created with inshape {} and outshape {} @ position {}"
              .format(self.inshape, self.outshape, position))

    def feedforward(self, questions):
        """
        Implementation of a max pooling layer.

        :param questions: numpy.ndarray, a batch of outputs from the previous layer
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

        self.output = self.excitation = \
            result.reshape([m] + list(self.outshape))
        return self.output

    def mp_feedforward(self, questions):
        return self.feedforward(questions)

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
        return deltas * prev.activation.derivative(prev.excitation)

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

        self.inputs = np.zeros(self.inshape)
        self.outshape = num_filters, self.outshape[0], self.outshape[1]
        self.filters = np.random.randn(num_filters, np.prod(fshape)) / np.sqrt(np.prod(inshape))
        print("<ConvLayer> created with inshape {} and outshape {} @ position {}"
              .format(self.inshape, self.outshape, position))

    def feedforward(self, questions):
        self.inputs = questions
        self.excitation = convolve(self.inputs, self.filters, mode="valid")
        self.output = self.activation(self.excitation)
        return self.output

    def old_feedforward(self, questions: np.ndarray):
        """
        Convolves the inputs with filters. Used in the learning phase.

        :param questions: numpy.ndarray, a batch of inputs. Shape should be (lessons, channels, x, y)
        :return: numpy.ndarray: outputs convolved with filters. Shape should be (lessons, filters, cx, cy)
        """
        self.inputs = questions

        # TODO: rethink this! Not working when channel > 1.
        recfields = np.array([[np.ravel(questions[qstn][:, start0:end0, start1:end1])
                              for start0, end0, start1, end1 in self.coords]
                             for qstn in range(questions.shape[0])])

        osh = [self.brain.m] + list(self.outshape)
        self.excitation = np.matmul(recfields, self.filters.T)
        self.excitation = np.transpose(self.excitation, (0, 2, 1)).reshape(osh)
        # self.excitation = np.transpose(np.inner(recfields, self.filters), axes=(0, 2, 1)).reshape(osh)
        self.output = self.activation(self.excitation)
        return self.output

    def mp_feedforward(self, questions: np.ndarray):
        """
        Convolves the inputs with filters. Used in the learning phase.

        :param questions: numpy.ndarray, a batch of inputs. Shape should be (lessons, channels, x, y)
        :return: numpy.ndarray: outputs convolved with filters. Shape should be (lessons, filters, cx, cy)
        """

        self.inputs = questions
        jobs = mp.cpu_count()
        mapool = mp.Pool(jobs)
        chunks = np.array_split(questions, jobs)
        recfields = mapool.map(self._getrfields, chunks)
        mapool.close()
        mapool.join()
        np.concatenate(recfields)

        self.excitation = np.transpose(np.inner(recfields, self.filters), axes=(0, 2, 1)
                                       ).reshape([self.brain.m] + list(self.outshape))
        self.output = self.activation(self.excitation)

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
        return deltas * prev.activation.derivative(prev.excitation)

    def receive_error(self, error_matrix: np.ndarray):
        """
        Fold the received error matrix.

        :param error_matrix: numpy.ndarray: backpropagated errors
        :return: None
        """
        self.error = error_matrix.reshape([self.brain.m] + list(self.outshape))

    def weight_update(self):
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
                    cvm_old = convolve(self.inputs[l][j], self.error[l][i], self.stride)
                    cvm = sigconvnd(self.inputs[l][j], self.error[l][i], mode="valid")
                    eq = np.equal(cvm, cvm_old)
                    # assert np.sum(eq) == eq.size
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


class FFLayer(_FCLayer):
    def __init__(self, brain, inputs, neurons, position, activation):
        _FCLayer.__init__(self, brain=brain,
                          neurons=neurons, position=position,
                          activation=activation)

        self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs)
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
        self.inputs = ravtm(questions)
        self.excitation = np.dot(self.inputs, self.weights) + self.biases
        self.output = self.activation(self.excitation)
        return self.output

    def mp_feedforward(self, questions):
        return self.feedforward(questions)

    def predict(self, questions):
        """
        Tranfsorms an input with the weights.

        This method has no side-effects. Used in prediction and testing.

        :param questions:
        :return:
        """
        return self.activation(np.dot(ravtm(questions), self.weights) + self.biases)

    def backpropagation(self):
        """
        Calculates the errors of the previous layer.

        :return: numpy.ndarray
        """
        prev = self.brain.layers[self.position-1]
        posh = [self.error.shape[0]] + list(prev.outshape)
        deltas = np.dot(self.error, self.weights.T).reshape(posh)
        return deltas * prev.activation.derivative(prev.excitation)

    def weight_update(self):
        """
        Updates the weight matrix with the calculated gradients

        :return: None
        """
        # Apply L2 regularization, aka weight decay
        l2 = l2term(self.brain.eta, self.brain.lmbd, self.brain.N)
        np.subtract(self.weights * l2,
                    np.dot(self.inputs.T, self.error) * (self.brain.eta / self.brain.m),
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


class DropOut(FFLayer):
    def __init__(self, brain, inputs, neurons, dropout, position, activation):
        FFLayer.__init__(self, brain, inputs, neurons, position, activation)

        self.dropchance = 1 - dropout
        self.mask = None

    def feedforward(self, questions):
        self.inputs = ravtm(questions)
        self.mask = np.random.uniform(0, 1, self.biases.shape) < self.dropchance
        self.excitation = (np.dot(self.inputs, self.weights) + self.biases) * self.mask
        self.output = self.activation(self.excitation)
        return self.output

    def predict(self, question):
        return FFLayer.predict(self, question) * self.dropchance

    def backpropagation(self):
        return FFLayer.backpropagation(self) * self.mask

    def weight_update(self):
        l2 = l2term(self.brain.eta, self.brain.lmbd, self.brain.N)
        np.subtract((self.weights * l2) * self.mask,
                    np.dot(self.inputs.T, self.error) * (self.brain.eta / self.brain.m),
                    out=self.weights)
        np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                    out=self.biases)


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
