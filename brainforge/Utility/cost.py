"""Cost function definitions"""

import numpy as np


class MSE:

    def __call__(self, outputs, targets):
        return 0.5 * np.linalg.norm(outputs - targets) ** 2

    def derivative(self, outputs, targets, excitations, activation):
        return activation.derivative(excitations) * np.subtract(outputs, targets)


class Xent:

    def __call__(self, outputs, targets):
        return np.sum(np.nan_to_num(
            np.subtract((0-targets) * np.log(outputs),
                        (1-targets) * np.log(1-outputs))))

    def derivative(self, outputs, targets, excitations=None, activation=None):
        return np.subtract(outputs, targets)


def mse(outputs, targets, excitations, activation_derivative):
    return activation_derivative(excitations) * np.subtract(outputs, targets)


def xent(outputs, targets, excitations=None, activation=None):
    return np.subtract(outputs, targets)


def loglikelyhood(outputs, targets=None, excitations=None, activation=None):
    return np.negative(np.max(np.log(outputs)))
