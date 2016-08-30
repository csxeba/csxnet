"""Cost function definitions"""
import abc

import numpy as np


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
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / a.shape[0]

    @staticmethod
    def derivative(outputs, targets):
        divtop = np.subtract(targets, outputs)
        divbot = np.subtract(outputs, 1.) * outputs
        d_xent = np.divide(divtop, divbot)
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


mse = _MSE()
xent = _Xent()
nll = _NLL()
cost = _Cost()
