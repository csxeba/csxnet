from csxnet import Network
from csxnet.brainforge.layers import DenseLayer, Experimental, Flatten

from csxdata import CData, roots


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.18)
    mnist.transformation = "std"
    return mnist


def build_cnn(data: CData):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestBrainforgeCNN")
    net.add(Experimental.ConvLayer(1, 4, 4))
    net.add(Flatten())
    net.add(DenseLayer(30, activation="sigmoid"))
    net.finalize("xent")
    return net


def xperiment():
    mnist = pull_mnist_data()
    net = build_cnn(mnist)

    net.fit(*mnist.table("learning"), batch_size=20,
            epochs=10, monitor=["acc"],
            validation=mnist.table("testing"))


if __name__ == '__main__':
    xperiment()
