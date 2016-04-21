import sys

import numpy as np

from datamodel import CData, mnist_to_lt
from brainforge.Architecture.NNModel import Network
from brainforge.Utility.cost import Xent, MSE
from brainforge.Utility.activations import Linear, Sigmoid, Tanh
# from thNets.thANN import ConvNet


datapath = "D:/Data/" if sys.platform == 'win32' else "/data/Prog/data/learning_tables/"
log = ""


def test_ANN():
    pca = 0
    data = CData(mnist_to_lt(datapath+"mnist.pkl.gz"),
                 cross_val=0.1, pca=pca)

    def get_FFNN():
        nw = Network(data, 0.5, 0.0, cost=Xent)
        nw.add_fc(1000, activation=Sigmoid)
        nw.add_drop(500, dropchance=0.5, activation=Sigmoid)
        nw.add_fc(250, activation=Sigmoid)
        nw.finalize_architecture(activation=Sigmoid)
        return nw

    def get_CNN():
        nw = Network(data, 0.5, 5.0, MSE)
        nw.add_conv()
        nw.add_pool()
        nw.finalize_architecture()
        return nw

    epochs = 20
    net = get_FFNN()

    for epoch in range(epochs):
        net.learn(20)
        ont, onl = net.evaluate(), net.evaluate("learning")
        print("Epoch {} Cost': {}".format(epoch+1, net.error))
        print("Acc: T: {} L: {}".format(ont, onl))
        global log
        log += "E:{}\nC:{}\nT:{}\nL:{}\n\n".format(epoch+1, net.error, ont, onl)


def xor():
    global log
    data = CData((np.array([[0, 0],[0, 1], [1, 0], [1, 1]]),
                np.array([[0], [1], [1], [0]])), .0, False)

    net = Network(data, 0.001, 0.0, 0.0, 0.0, MSE)
    net.add_fc(4)
    net.add_fc(4)
    net.finalize_architecture()

    cost = MSE()
    for epoch in range(1, 10):
        net.learn(4)
        c = cost(net.predict(data.data), data.indeps)
        log += "C@{}: {}\n".format(epoch, c)
        if epoch % 1 == 0:
            print("Cost @ {}: {}".format(epoch, c))




def test_thCNN():
    # lt = unpickle_gzip(datapath+"learning_tables/mnist.pkl.gz")
    # lt = (lt[0]/255, lt[1])
    from datamodel import mnist_to_lt
    lt = mnist_to_lt(datapath+"learning_tables/mnist.pkl.gz")
    data = CData(lt, cross_val=0.2)
    # myData = CData(lt)
    # myData.myData = myData.myData.reshape(osh[0], 1, osh[1], osh[2])
    # myData.split_data()
    network = ConvNet(data, eta=0.15, lmbd=0.0)
    for epoch in range(30):
        network.learn(10)
        tcost, tacc = network.evaluate(on="testing")
        # lcost, lacc = self.evaluate(on="learning")
        print("Done {} epochs!".format(epoch + 1))
        print("Cost on T: {}\nAcc on T: {}".format(tcost, tacc))
        global log
        log += "E:{}\nC:{}\nT:{}\nL:{}\n\n".format(epoch + 1, tcost, tacc, "(not measured)")


if __name__ == '__main__':
    xor()
    logfl = open("log.txt", "w")
    logfl.write(log)
    logfl.close()
