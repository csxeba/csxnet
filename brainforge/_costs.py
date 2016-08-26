"""Cost function definitions"""
import abc

import numpy as np


class _CostFnBase(abc.ABC):

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    def derivative(outputs, targets, activation): pass


class MSE(_CostFnBase):

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    @staticmethod
    def derivative(outputs, targets, activation):
        return activation.derivative(outputs) * np.subtract(outputs, targets)

    def __str__(self):
        return "MSE"


class Xent(_CostFnBase):

    def __call__(self, outputs, targets):
        return np.sum(np.nan_to_num(
            np.subtract((0-targets) * np.log(outputs),
                        (1-targets) * np.log(1-outputs))))

    @staticmethod
    def derivative(outputs, targets, activation=None):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "Xent"


class NLL(_CostFnBase):
    """Negative log-likelyhood cost function"""
    def __call__(self, outputs, targets=None):
        return np.negative(np.log(outputs))

    @staticmethod
    def derivative(outputs, targets, activation):
        pass

    def __str__(self):
        return "NLL"


class _Cost:
    @staticmethod
    def mse():
        return MSE()

    @staticmethod
    def xent():
        return Xent()

    @staticmethod
    def nll():
        return NLL()

    def __getitem__(self, item: str):
        if not isinstance(item, str):
            raise TypeError("Please supply a string!")
        item = item.lower()
        d = {"mse": MSE,
             "xent": Xent,
             "nll": NLL}
        if item not in d:
            raise IndexError("Requested cost function is unsupported!")
        return d[item]()


mse = MSE()
xent = Xent()
nll = NLL()
cost = _Cost()
