import abc

import numpy as np


class _ActivationFunctionBase(abc.ABC):
    def __call__(self, Z: np.ndarray): pass

    def __str__(self): return ""

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
        return np.greater(0.0, A).astype("float32")


class _SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.divide(Z, np.sum(Z, axis=0))

    def __str__(self): return "softmax"

    @staticmethod
    def derivative(A):
        # This is the negative of the outer product of the last axis with itself
        J = - A[:, :, None] * A[:, None, :]  # given by -a_i*a_j, where i =/= j
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


sigmoid = _Sigmoid()
tanh = _Tanh()
linear = _Linear()
relu = _ReLU()
softmax = _SoftMax()
activation = _Activation()
