import abc

import numpy as np


def l1term(eta, lmbd, N):
    return (eta * lmbd) / N


def l2term(eta, lmbd, N):
    return 1 - ((eta * lmbd) / N)


def outshape(inshape: tuple, fshape: tuple, stride: int):
    """
    Calculates the shape of an output matrix if a filter of shape
    <fshape> gets slided along a matrix of shape <fanin> with a
    stride of <stride>.
    Returns x, y sizes of the output matrix
    """
    output = [int((x - ins) / stride) + 1 if (x - ins) % stride == 0 else "NaN"
              for x, ins in zip(inshape[1:3], fshape[1:3])]
    if "NaN" in output:
        raise RuntimeError("Shapes not compatible!")
    return tuple(output)


def calcsteps(inshape: tuple, fshape: tuple, stride: int):
    """
    Calculates the coordinates required to slide
    a filter of shape <fshape> along a matrix of shape <inshape>
    with a stride of <stride>.
    Returns a list of coordinates
    """
    xsteps, ysteps = outshape(inshape, fshape, stride)

    startxes = np.arange(xsteps) * stride
    startys = np.arange(ysteps) * stride

    endxes = startxes + fshape[1]
    endys = startys + fshape[2]

    coords = []

    for sy, ey in zip(startys, endys):
        for sx, ex in zip(startxes, endxes):
            coords.append((sx, ex, sy, ey))

    return tuple(coords)


def numerical_gradients(network, X, y, epsilon=1e-5):
    ws = network.get_weights(unfold=True)
    numgrads = np.zeros_like(ws)
    perturb = np.copy(numgrads)

    nparams = ws.size
    print("Calculating numerical gradients...")
    for i in range(nparams):
        print("\r{0:>{1}} / {2:<}".format(i+1, len(str(nparams)), nparams), end=" ")
        perturb[i] += epsilon

        network.set_weights(ws + perturb, fold=True)
        pred1 = network.prediction(X)
        cost1 = network.cost(pred1, y)
        network.set_weights(ws - perturb, fold=True)
        pred2 = network.prediction(X)
        cost2 = network.cost(pred2, y)

        numgrads[i] = (cost1 - cost2)
        perturb[i] = 0.0

    numgrads /= (2 * epsilon)
    network.set_weights(ws, fold=True)

    print("Done!")

    return numgrads


def analytical_gradients(network, X, y):
    anagrads = np.zeros((network.nparams,))
    network.prediction(X)
    network.backpropagation(y)

    start = 0
    for layer in network.layers:
        if not layer.trainable:
            continue
        end = start + layer.nparams
        anagrads[start:end] = layer.gradients
        start = end

    return anagrads


def gradient_check(network, X, y, epsilon=1e-5, display=False, verbose=1):

    def fold_difference_matrices(dvec):
        diffs = []
        start = 0
        for layer in network.layers[1:]:
            weight = start + layer.weights.size
            bias = weight + layer.biases.size
            diffs.append(dvec[start:weight].reshape(layer.weights.shape))
            diffs.append(dvec[weight:bias].reshape(layer.biases.shape))
            start = bias
        return diffs

    def analyze_difference_matrices(dvec):
        from matplotlib import pyplot as plt
        dmats = fold_difference_matrices(dvec)
        for i, d in enumerate(dmats):
            print("Sum of difference matrix no {0}: {1:.4e}".format(i, d.sum()))
            plt.matshow(np.atleast_2d(np.abs(d)))
            plt.show()

    def get_results(er):
        if relative_error < 1e-7:
            errcode = 0
        elif relative_error < 1e-5:
            errcode = 1
        elif relative_error < 1e-3:
            errcode = 2
        else:
            errcode = 3

        if verbose:
            print("Result of gradient check:")
            print(["Gradient check passed, error {} < 1e-7",
                   "Suspicious gradients, 1e-7 < error {} < 1e-5",
                   "Gradient check failed, 1e-5 < error {} < 1e-3",
                   "Fatal fail in gradient check, error {} > 1e-3"
                   ][errcode].format("({0:.1e})".format(er)))

        return True if errcode < 3 else False

    norm = np.linalg.norm
    analytic = analytical_gradients(network, X, y)
    numeric = numerical_gradients(network, X, y, epsilon=epsilon)
    diff = analytic - numeric
    relative_error = norm(diff) / max(norm(numeric), norm(analytic))

    passed = get_results(relative_error)

    if display:
        analyze_difference_matrices(diff)

    return passed


def white(*dims):
    """Returns a white noise tensor"""
    return np.random.randn(*dims) / np.sqrt(dims[0] / 2.)


def white_like(array):
    return white(*array.shape)


class _ActivationFunctionBase(abc.ABC):
    def __call__(self, Z: np.ndarray): pass

    def __str__(self): raise NotImplementedError

    @staticmethod
    def derivative(Z: np.ndarray): pass


class _Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    @staticmethod
    def derivative(A):
        return np.multiply(A, np.subtract(1.0, A))


class _Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    @staticmethod
    def derivative(A):
        return np.subtract(1.0, np.square(A))


class _Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    @staticmethod
    def derivative(Z):
        return np.ones_like(Z)


class _ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    @staticmethod
    def derivative(A):
        d = np.greater(A, 0.0).astype("float32")
        return d


class _SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    @staticmethod
    def derivative(A: np.ndarray):
        # This is the negative of the outer product of the last axis with itself
        J = A[..., None] * A[:, None, :]  # given by -a_i*a_j, where i =/= j
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = A * (1. - A)  # given by a_i(1 - a_j), where i = j
        return J.sum(axis=1)  # sum for each sample


class _Activation:

    @property
    def sigmoid(self):
        return _Sigmoid()

    @property
    def tanh(self):
        return _Tanh()

    @property
    def linear(self):
        return _Linear()

    @property
    def relu(self):
        return _ReLU()

    @property
    def softmax(self):
        return _SoftMax()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {str(fn).lower(): fn for fn in (_Sigmoid(), _Tanh(), _Linear(), _ReLU(), _SoftMax())}
        if item not in d:
            raise IndexError("Requested activation function ({}) is unsupported!".format(item))
        return d[item]


class _CostFnBase(abc.ABC):

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    def derivative(outputs, targets): pass


class _MSE(_CostFnBase):

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    @staticmethod
    def derivative(outputs, targets):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "MSE"


class _Xent(_CostFnBase):

    def __call__(self, a: np.ndarray, y: np.ndarray):
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    @staticmethod
    def derivative(outputs, targets):
        enum = np.subtract(targets, outputs)
        denom = np.subtract(outputs, 1.) * outputs
        d_xent = np.divide(enum, denom)
        return d_xent

    def __str__(self):
        return "Xent"


class _NLL(_CostFnBase):
    """Negative logstring-likelyhood cost function"""
    def __call__(self, outputs, targets=None):
        return np.negative(np.log(outputs))

    @staticmethod
    def derivative(outputs, targets=None):
        return np.reciprocal(outputs)

    def __str__(self):
        return "NLL"


class _Cost:
    @property
    def mse(self):
        return _MSE()

    @property
    def xent(self):
        return _Xent()

    @property
    def nll(self):
        return _NLL()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {str(fn).lower(): fn for fn in (_NLL(), _MSE(), _Xent())}
        if item not in d:
            raise IndexError("Requested cost function is unsupported!")
        return d[item]


sigmoid = _Sigmoid()
tanh = _Tanh()
linear = _Linear()
relu = _ReLU()
softmax = _SoftMax()
act_fns = _Activation()

mse = _MSE()
xent = _Xent()
nll = _NLL()
cost_fns = _Cost()
