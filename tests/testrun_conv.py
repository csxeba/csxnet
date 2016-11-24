from csxnet import Network
from csxnet.brainforge.layers import DenseLayer, Experimental

from csxdata import CData, roots


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=1.8)
    mnist.transformation = "std"


def build_cnn(data: CData):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestBrainforgeCNN")
    net.add(Experimental.ConvLayer(1, 4, 4))
    net.add(DenseLayer(30, activation="sigmoid"))
    net.finalize("xent")


def xperiment():
    mnist = pull_mnist_data()
    net = build_cnn(mnist)


