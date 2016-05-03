import numpy as np


class _ActivationFunctionBase(object):
    def __call__(self, Z: np.ndarray): pass

    def __str__(self): return ""

    @staticmethod
    def derivative(Z: np.ndarray): pass


class Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    @staticmethod
    def derivative(Z):
        return np.multiply(Sigmoid()(Z), np.subtract(1.0, Sigmoid()(Z)))


class FastSigmoid(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.divide(Z, np.add(1, np.abs(Z)))

    def __str__(self): return "fast sigmoid"

    @staticmethod
    def derivative(Z):
        return np.divide(1.0, np.add(np.abs(Z), 1.0)**2)


class Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    @staticmethod
    def derivative(Z):
        return np.subtract(1.0, np.square(Tanh()(Z)))


class Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    @staticmethod
    def derivative(Z):
        return np.ones_like(Z)


class ReL(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    @staticmethod
    def derivative(Z):
        return np.greater(0.0, Z).astype("float64")
