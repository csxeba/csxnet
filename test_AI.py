import sys

from datamodel import CData
from brainforge.Architecture.NNModel import Network
from brainforge.Utility.cost import Xent
from brainforge.Utility.activations import Linear, Sigmoid, ReL
from thNets.thANN import ConvNet


datapath = "D:/Data/" if sys.platform == 'win32' else "/data/Prog/data/"
log = ""


def test_ANN():
    pca = 0
    data = CData(datapath + "learning_tables/mnist.pkl.gz",
                 cross_val=0.1, pca=pca)

    def get_FFNN():
        nw = Network(data, 2.0, 0.0, cost=Xent)
        # nw.add_fc(120, activation=Sigmoid)
        nw.add_drop(120, dropchance=0.5, activation=Sigmoid)
        nw.finalize_architecture(activation=Sigmoid)
        return nw

    def get_CNN():
        nw = Network(data, 0.5, 5.0, Xent)
        nw.add_conv()
        nw.add_pool()
        nw.finalize_architecture()
        return nw

    epochs = 30
    net = get_FFNN()

    for epoch in range(epochs):
        net.learn(10)
        ont, onl = net.evaluate(), net.evaluate(("learning"))
        print("Epoch {} Cost': {}".format(epoch+1, net.error))
        print("Acc: T: {} L: {}".format(ont, onl))
        global log
        log += "E:{}\nC:{}\nT:{}\nL:{}\n\n".format(epoch+1, net.error, ont, onl)


def test_thCNN():
    # lt = unpickle_gzip(datapath+"learning_tables/mnist.pkl.gz")
    # lt = (lt[0]/255, lt[1])
    data = CData(datapath+"learning_tables/mnist.pkl.gz")
    # myData = CData(lt)
    osh = data.data.shape
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
    test_thCNN()
    logfl = open("log.txt", "w")
    logfl.write(log)
    logfl.close()
