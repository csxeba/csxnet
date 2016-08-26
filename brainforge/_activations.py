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
    def derivative(A):
        return np.multiply(A, np.subtract(1.0, A))


class Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    @staticmethod
    def derivative(A):
        return np.subtract(1.0, np.square(A))


class Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    @staticmethod
    def derivative(Z):
        return np.ones_like(Z)


class ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    @staticmethod
    def derivative(A):
        return np.greater(0.0, A).astype("float32")


class SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.divide(Z, np.sum(Z, axis=0))

    def __str__(self): return "softmax"

    @staticmethod
    def derivative(A):
        raise NotImplementedError("Sorry for this...")


class _Activation:

    @staticmethod
    def sigmoid():
        return Sigmoid()

    @staticmethod
    def tanh():
        return Tanh()

    @staticmethod
    def linear():
        return Linear()

    @staticmethod
    def relu():
        return ReLU()

    @staticmethod
    def softmax():
        return SoftMax()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {"sigmoid": Sigmoid,
             "tanh": Tanh,
             "linear": Linear,
             "relu": ReLU,
             "softmax": SoftMax}
        if item not in d:
            raise IndexError("Requested activation function is unsupported!")
        return d[item]()


sigmoid = Sigmoid()
tanh = Tanh()
linear = Linear()
relu = ReLU()
softmax = SoftMax()
activation = _Activation()
