import abc

import numpy as np

from csxnet.datamodel import _Data as DataWrapper


class NeuralNetworkBase(abc.ABC):

    def __init__(self, data: DataWrapper,
                 eta: float, lmbd1: float, lmbd2, mu: float,
                 cost: callable):
        # Referencing the data wrapper on which we do the learning
        self.data = data
        self.N = data.N
        self.fanin, self.outsize = data.neurons_required()

        # Parameters required for SGD
        self.eta = eta
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.mu = mu

        # Containers and self-describing variables
        self.layers = []
        self.architecture = []
        self.age = 0
        self.name = ""

    @abc.abstractmethod
    def learn(self, batch_size: int): pass

    @abc.abstractmethod
    def evaluate(self, on: str): pass

    @abc.abstractmethod
    def predict(self, questions: np.ndarray): pass

    @abc.abstractmethod
    def describe(self): pass
