from csxnet.model import Network

from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable

mnistpath = roots["misc"] + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    data = CData(mnist_tolearningtable(path))
    data.learning += 1e-6
    return data


def get_dense_network(data):
    nw = Network(data, 0.5, 0.0, 5.0, 0.0, cost="xent")
    nw.add_fc(60, activation="sigmoid")
    nw.finalize_architecture(activation="sigmoid")
    return nw


def test_ann():
    net = get_dense_network(get_mnist_data(mnistpath))
    dsc = net.describe()
    log(dsc)
    print(dsc)
    net.fit(batch_size=20, epochs=30, verbose=1, monitor=["acc"])


if __name__ == '__main__':
    log(" --- CsxNet Brainforge testrun ---")
    test_ann()
    log(logstring)
