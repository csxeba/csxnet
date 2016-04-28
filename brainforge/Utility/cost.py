"""Cost function definitions"""

import numpy as np


class _CostFnBase(object):

    def __call__(self, outputs, targets): pass

    def __str__(self): return ""

    @staticmethod
    def derivative(outputs, targets, excitations, activation): pass


class MSE(_CostFnBase):

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    @staticmethod
    def derivative(outputs, targets, excitations, activation):
        return activation.derivative(excitations) * np.subtract(outputs, targets)

    def __str__(self):
        return "MSE"


class Xent(_CostFnBase):

    def __call__(self, outputs, targets):
        return np.sum(np.nan_to_num(
            np.subtract((0-targets) * np.log(outputs),
                        (1-targets) * np.log(1-outputs))))

    @staticmethod
    def derivative(outputs, targets, excitations=None, activation=None):
        return np.subtract(outputs, targets)

    def __str__(self):
        return "Xent"


def mse(outputs, targets, excitations, activation_derivative):
    return activation_derivative(excitations) * np.subtract(outputs, targets)


def xent(outputs, targets, excitations=None, activation=None):
    return np.subtract(outputs, targets)


def loglikelyhood(outputs, targets=None, excitations=None, activation=None):
    return np.negative(np.max(np.log(outputs)))
