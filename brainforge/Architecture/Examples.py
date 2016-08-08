from NNModel import Network
from brainforge.activations import *
from brainforge.cost import *
from .FFNN import FFLayerBrain
from ..Utility.DataModel import CData


class OnMNIST:

    # Data is pulled in as a class variable. This might not be the best idea ->
    # would this lead to duplication in memory? IDK whether this ever moves out
    # of scope...
    data = CData("D:/Data/mnist.pkl.gz", cross_val=1-0.8571428571428571)

    @staticmethod
    def shallow_CNN():
        eta = 0.8
        lmbda = 5.0
        cost = Xent

        net = Network(OnMNIST.data, eta, lmbda, cost)
        net.add_conv(fshape=(3, 3), n_filters=1, stride=1, activation=Sigmoid)
        net.add_pool(pool=2)
        net.add_fc(neurons=80, activation=Sigmoid)
        net.finalize_architecture(activation=Sigmoid)

        epochs = 10

        return net, epochs

    @staticmethod
    def deep_CNN():
        eta = 0.8
        lmbda = 5.0
        cost = Xent

        net = Network(OnMNIST.data, eta, lmbda, cost)

        net.add_conv(fshape=(9, 9), n_filters=3, stride=1, activation=Sigmoid)
        net.add_pool(pool=2)
        net.add_conv(fshape=(3, 3), n_filters=3, stride=1, activation=Sigmoid)
        net.add_pool(pool=2)
        net.finalize_architecture(activation=Sigmoid)

        epochs = 10

        return net, epochs

    @staticmethod
    def from_FFClass():
        eta = 0.8
        lmbda = 5.0
        cost = Xent

        epochs = 30

        net = FFLayerBrain(hiddens=(80,), data=OnMNIST.data, eta=eta, lmbd=lmbda, cost=cost, activation=Sigmoid)

        return net, epochs

    @staticmethod
    def dropper():
        eta = 0.8
        lmbda = 5.0
        cost = Xent

        net = Network(OnMNIST.data, eta=eta, lmbd=lmbda, cost=cost)
        net.add_drop(60)
        net.finalize_architecture()

        epochs = 30

        return net, epochs

    @staticmethod
    def by_Nielsen():
        """Nielsen's architecture"""
        eta = 0.5
        lmbda = 5.0
        cost = Xent
        epochs = 30

        net = Network(OnMNIST.data, eta, lmbda, cost)

        net.add_fc(30, activation=Sigmoid)
        net.finalize_architecture(activation=Sigmoid)

        return net, epochs

    @staticmethod
    def by_Misi():
        """Misi's architecture"""
        eta = 3.0
        lmbda = 0.0
        cost = MSE

        net = Network(OnMNIST.data, eta, lmbda, cost)

        net.add_fc(30)
        net.finalize_architecture()

        epochs = 5

        return net, epochs
