import sys
import time

from brainforge.Architecture.NNModel import Network
from brainforge.Utility.activations import *
from brainforge.Utility.cost import Xent, MSE
from datamodel import CData, mnist_to_lt


datapath = "D:/Data/" if sys.platform == 'win32' else "/data/Prog/data/learning_tables/"
log = ""


def test_ANN():
    pca = 0
    data = CData(mnist_to_lt(datapath+"mnist.pkl.gz"),
                 cross_val=0.1, pca=pca)

    def get_FFNN():
        nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=Xent)
        nw.add_fc(120, activation=Sigmoid)
        nw.add_fc(60, activation=Sigmoid)
        nw.finalize_architecture(activation=Sigmoid)
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

    net = Network(data, 0.01, 0.0, 0.001, 0.1, MSE)
    net.add_fc(6, activation=Tanh)
    net.finalize_architecture()

    cost = MSE()
    epoch = 1
    while 1:
        net.learn(batch_size=1)
        c = cost(net.predict(data.data), data.indeps)
        log += "C@{}: {}\n".format(epoch, c)
        if epoch % 1 == 0:
            print("Cost @ {}: {}".format(epoch, c))
            print("Prediction:")
            for i in range(4):
                print(data.data[i], "is", net.predict(data.data[i]), "real:", data.indeps[i])
        epoch += 1
        if c < 0.01:
            break
        time.sleep(0.5)


if __name__ == '__main__':
    test_ANN()
    logfl = open("log.txt", "w")
    logfl.write(log)
    logfl.close()
