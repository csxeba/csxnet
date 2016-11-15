from csxnet.ann import Network

from csxdata import CData, roots, log
from csxdata.utilities.parsers import mnist_tolearningtable

mnistpath = roots["misc"] + "mnist.pkl.gz"
logstring = ""


def get_mnist_data(path):
    data = CData(mnist_tolearningtable(path, fold=False))
    return data


def get_dense_network(data):
    nw = Network(data, 0.5, 0.0, 0.0, 0.0, cost="xent")
    nw.add_fc(30, activation="sigmoid")
    nw.finalize_architecture(activation="sigmoid")
    return nw


def test_ann():

    def perform_gradient_checking():
        from csxnet.util import gradient_check

        net.fit(20, 1, verbose=1)
        logstr = "Gradient checked performed and "
        if gradient_check(net, *net.data.table("testing", m=1000), display=False):
            logstr += "passed!"
        else:
            logstr += "FAILED!"
        log(logstr)
        print(logstring)

    log(" --- CsxNet Brainforge testrun ---")
    net = get_dense_network(get_mnist_data(mnistpath))
    dsc = net.describe()
    log(dsc)
    print(dsc)

    perform_gradient_checking()

    net.fit(batch_size=20, epochs=30, verbose=1, monitor=["acc"])
    log(" --- End of CsxNet testrun ---")


if __name__ == '__main__':
    test_ann()
