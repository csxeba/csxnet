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


def mse(outputs, targets, excitations, activation_derivative):
    return activation_derivative(excitations) * np.subtract(outputs, targets)


def xent(outputs, targets, excitations=None, activation=None):
    del excitations, activation
    return np.subtract(outputs, targets)


fromstring = {"xent": Xent,
              "mse": MSE}
