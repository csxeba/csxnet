import sys
import time

from brainforge.Architecture.NNModel import Network
from brainforge.Utility.activations import *
from brainforge.Utility.cost import Xent, MSE
from datamodel import CData, mnist_to_lt


datapath = "D:/Data/misc/" if sys.platform == 'win32' else "/data/Prog/data/misc/"
log = ""


def test_ANN():
    pca = 0
    lt = mnist_to_lt(datapath+"mnist.pkl.gz")
    data = CData(lt, cross_val=0.1, pca=pca)

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


def keras_vs_mnist():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation

    data = CData(mnist_to_lt(datapath + "mnist.pkl.gz", reshape=False))
    fanin, outshape = data.neurons_required()

    net = Sequential()
    net.add(Dense(output_dim=120, input_dim=np.prod(fanin)))
    net.add(Activation("tanh"))
    net.add(Dense(output_dim=outshape))
    net.add(Activation("softmax"))

    net.compile(optimizer="sgd", loss="categorical_crossentropy")

    learnX, learnY = data.table("learning")
    testX, testY = data.table("testing", False)
    net.fit(learnX, learnY, batch_size=10, nb_epoch=30, validation_data=(testX, testY),
            show_accuracy=True)


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
    keras_vs_mnist()
    # logfl = open("log.txt", "w")
    # logfl.write(log)
    # logfl.close()
