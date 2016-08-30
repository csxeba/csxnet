from csxnet.model import Network
from csxnet.brainforge import activations
from csxnet.brainforge import costs

from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable


datapath = roots["misc"]
mnistpath = datapath + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    lt = mnist_tolearningtable(path)
    data = CData(lt, cross_val=0.18)
    data.transformation = "standardization"
    return data


def get_dense_network(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost="mse")
    nw.add_fc(120, activation=activations.tanh)
    nw.add_fc(60, activation=activations.tanh)
    nw.finalize_architecture(activation=activations.sigmoid)
    return nw


def get_dropout_network(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=costs.xent)
    nw.add_drop(120, dropchance=0.5, activation=activations.tanh)
    nw.add_fc(60, activation=activations.tanh)
    nw.finalize_architecture(activation=activations.sigmoid)
    return nw


def get_cnn(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost=costs.xent)
    nw.add_conv((5, 5), 1, 1, activation=activations.tanh)
    nw.add_pool(pool=2)
    nw.add_fc(60, activation=activations.tanh)
    nw.finalize_architecture(activation=activations.sigmoid)
    return nw


def test_ann(architecture):
    net = {"f": get_dense_network,
           "c": get_cnn,
           "d": get_dropout_network}[
        architecture[0].lower()](get_mnist_data(mnistpath))

    net.describe(verbose=True)

    epochs = 10
    for epoch in range(epochs):
        net.learn(50)
        (_, acct), (cost, accl) = net.evaluate(), net.evaluate("learning")
        # print("Epoch {} Cost: {}".format(epoch+1, cost))
        print("Acc: T: {} L: {}".format(acct, accl))
        global logstring
        logstring += "E:{}\nC:{}\nT:{}\nL:{}\n\n".format(epoch + 1, cost, acct, accl)


if __name__ == '__main__':
    log(" --- CsxNet Brainforge testrun ---")
    test_ann("FF")
    log(logstring)
