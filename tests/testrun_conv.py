from csxnet import Network
from csxnet.brainforge.layers import DenseLayer, Experimental, Flatten

from csxdata import CData, roots


def pull_mnist_data():
    mnist = CData(roots["misc"] + "mnist.pkl.gz", cross_val=0.18)
    mnist.transformation = "std"
    return mnist


def build_keras_reference(data: CData):
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Flatten
    inshape, outshape = data.neurons_required
    net = Sequential([
        Conv2D(nb_filter=1, nb_row=3, nb_col=5, activation="tanh",
               input_shape=inshape),
        Flatten(),
        Dense(outshape[0], activation="sigmoid")
    ])
    net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return net


def build_cnn(data: CData):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestBrainforgeCNN")
    net.add(Experimental.ConvLayer(3, 3, 5, activation="tanh"))
    net.add(Flatten())
    net.add(DenseLayer(outshape, activation="sigmoid", trainable=False))
    net.finalize("xent")
    return net


def xperiment():
    mnist = pull_mnist_data()
    net = build_cnn(mnist)
    net.gradient_check(*mnist.table("testing", m=10))

    net.fit(*mnist.table("learning"), batch_size=20, epochs=10, monitor=("acc",),
            validation=mnist.table("testing"))


if __name__ == '__main__':
    xperiment()
