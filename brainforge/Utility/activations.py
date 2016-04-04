import numpy as np


class _ActivationFunctionBase:
    def __call__(self, Z: np.ndarray): pass

    def __str__(self): return ""

    def derivative(self, Z: np.ndarray): pass


class Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, Z):
        return np.multiply(self(Z), np.subtract(1.0, self(Z)))


class FastSigmoid(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.divide(Z, np.add(1, np.abs(Z)))

    def __repr__(self): return "fast sigmoid"

    def derivative(self, Z):
        return np.divide(1.0, np.add(np.abs(Z), 1.0)**2)


class Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, Z):
        return np.subtract(1.0, np.square(self(Z)))


class Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z):
        return np.ones_like(Z)


class ReL(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "rectified linear"

    def derivative(self, Z):
        return np.greater(0.0, Z).astype(float)
