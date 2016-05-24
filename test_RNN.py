from datamodel import CData
from brainforge.Architecture.NNModel import RNN
from brainforge.Utility.cost import Xent
from brainforge.Utility.activations import *

eta = 0.3
hidden_neurons = 3
time = 7
source = ""


def getrnn(neurons):
    data = CData(source, cross_val=0.2, header=False)
    data.standardize()
    net = RNN(neurons, data, eta, Xent, Tanh)
    return net


def main():
    rnn = getrnn(hidden_neurons)
    rnn.learn(time)
