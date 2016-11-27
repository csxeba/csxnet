import abc

import numpy as np


class _OpBase(abc.ABC):

    def __init__(self):
        self._inshape = None
        self._outshape = None

    @property
    @abc.abstractmethod
    def outshape(self):
        raise NotImplementedError

    @property
    def inshape(self):
        return self._inshape

    @inshape.setter
    def inshape(self, shape):
        self._inshape = shape

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def backwards(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Flatten(_OpBase):

    def __init__(self):
        from ..util import rtm
        _OpBase.__init__(self)
        self.op = rtm

    def __str__(self):
        return "Flatten"

    def __call__(self, A):
        return self.op(A)

    def backwards(self, A):
        m = A.shape[0]
        return A.reshape(m, *self.inshape)

    @property
    def outshape(self):
        return np.prod(self.inshape),


class Convolution(_OpBase):
    def __init__(self, mode="valid"):
        from ..util import convolve
        _OpBase.__init__(self)
        if mode not in ("valid", "full"):
            raise ValueError("Only valid and full convolution is supported, not {}".format(mode))
        self.mode = mode
        self.op = convolve

    def valid(self, A, F):
        return self.op(A, F, stride=1)

    def full(self, A, F):
        fx, fy = F.shape[:2]
        px, py = fx - 1, fy - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (px, px), (py, py)),
                   mode="constant", constant_values=0.)
        return self.valid(pA, F)

    def __call__(self, A, F):
        return self.valid(A, F) if self.mode == "valid" else self.full(A, F)

    def backwards(self, A, F):
        return self.valid(A, F) if self.mode == "full" else self.full(A, F)

    @property
    def inshape(self):
        return self._inshape

    @inshape.setter
    def inshape(self, shape):
        _OpBase.inshape = shape

    @property
    def outshape(self):
        return self._outshape

    def __str__(self):
        return "Convolution"


class _ActivationFunctionBase(_OpBase):

    def __call__(self, Z: np.ndarray): pass

    def __str__(self): raise NotImplementedError

    def derivative(self, Z: np.ndarray):
        raise NotImplementedError

    def backwards(self, A):
        return self.derivative(A)

    @property
    def outshape(self):
        return self._inshape


class _Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, A):
        return np.multiply(A, np.subtract(1.0, A))


class _Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, A):
        return np.subtract(1.0, np.square(A))


class _Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z):
        return np.ones_like(Z)


class _ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    def derivative(self, A):
        d = np.greater(A, 0.0).astype("float32")
        return d


class _SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    def derivative(self, A: np.ndarray):
        # This is the negative of the outer product of the last axis with itself
        J = A[..., None] * A[:, None, :]  # given by -a_i*a_j, where i =/= j
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = A * (1. - A)  # given by a_i(1 - a_j), where i = j
        return J.sum(axis=1)  # sum for each sample


sigmoid = _Sigmoid()
tanh = _Tanh()
linear = _Linear()
relu = _ReLU()
softmax = _SoftMax()

act_fns = {key.lower(): fn for key, fn in locals().items() if key[0] not in ("_", "F")}
