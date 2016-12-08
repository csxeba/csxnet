from csxnet import Network
from csxnet.brainforge.layers import DenseLayer, DropOut, HighwayLayer, HighwayAlt

from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable

mnistpath = roots["misc"] + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    data = CData(mnist_tolearningtable(path, fold=False))
    return data


def get_dense_network(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestDenseNet")
    nw.add(DenseLayer(30, activation="sigmoid"))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize("mse", optimizer="sgd")
    return nw


def get_drop_net(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestDropoutNet")
    nw.add(DenseLayer(30, activation="sigmoid"))
    nw.add(DropOut(0.5))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize("mse", optimizer="sgd")
    return nw


def get_highway_net(data):
    fanin, fanout = data.neurons_required
    nw = Network(fanin, name="TestHighwayNet")
    nw.add(DenseLayer(30, activation="tanh"))
    nw.add(HighwayAlt(activation="tanh"))
    nw.add(HighwayAlt(activation="tanh"))
    nw.add(DenseLayer(fanout, activation="sigmoid"))
    nw.finalize(cost="xent", optimizer="sgd")
    return nw


def test_ann():

    log(" --- CsxNet Brainforge testrun ---")
    mnist = get_mnist_data(mnistpath)
    net = get_highway_net(mnist)
    dsc = net.describe()
    log(dsc)
    print(dsc)

    net.fit(*mnist.table("learning", m=20), batch_size=20, epochs=1, verbose=0, shuffle=False)
    if not net.gradient_check(*mnist.table("testing", shuff=False, m=20), verbose=1):
        log("Gradient check failed!")
    else:
        log("Gradient check passed!")

    net.fit_csxdata(mnist, batch_size=20, epochs=30, verbose=1, monitor=["acc"])
    log(net.describe(0))
    log(" --- End of CsxNet testrun ---")


if __name__ == '__main__':
    test_ann()
