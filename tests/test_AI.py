from csxnet.model import Network
from csxnet.brainforge.activations import *
from csxnet.brainforge.cost import Xent

from csxdata.frames import CData
from csxdata.utilities.const import roots


datapath = roots["misc"]
mnistpath = datapath + "mnist.pkl.gz"
log = ""


def get_mnist_data(path):
    pca = 0
    lt = mnist_to_lt(path)
    data = CData(lt, cross_val=0.18, pca=pca)
    data.self_standardize()
    return data


def get_dense_network(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=Xent)
    nw.add_fc(120, activation=Tanh)
    nw.add_fc(60, activation=Tanh)
    nw.finalize_architecture(activation=Sigmoid)
    return nw


def get_dropout_network(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=Xent)
    nw.add_drop(120, dropchance=0.5, activation=Tanh)
    nw.add_fc(60, activation=Tanh)
    nw.finalize_architecture(activation=Sigmoid)
    return nw


def get_cnn(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=Xent)
    nw.add_conv((5, 5), 1, 1, activation=Tanh)
    nw.add_pool(pool=2)
    nw.add_fc(60, activation=Tanh)
    nw.finalize_architecture(activation=Sigmoid)
    return nw


def test_ANN(net="FF"):

    net = {"f": get_dense_network,
           "c": get_cnn,
           "d": get_dropout_network}[
        net[0].lower()](get_mnist_data(mnistpath))

    net.describe(verbose=True)

    epochs = 10
    for epoch in range(epochs):
        net.learn(20)
        ont, onl = net.evaluate(), net.evaluate("learning")
        print("Epoch {} Cost': {}".format(epoch+1, net.error))
        print("Acc: T: {} L: {}".format(ont, onl))
        global log
        log += "E:{}\nC:{}\nT:{}\nL:{}\n\n".format(epoch+1, net.error, ont, onl)

if __name__ == '__main__':
    test_ANN("D")
    with open("./logs/testDropOut_log.txt", "w") as outfl:
        outfl.write(log)
        outfl.close()
