from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me

from csxnet import Network
from csxnet.brainforge.layers import RLayer, LSTM, DenseLayer, EchoLayer

TIMESTEP = 10
NGRAM = 1


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=NGRAM, timestep=TIMESTEP,
                    cross_val=0.01)


def build_reference_network(data: Sequence):
    from keras.models import Sequential
    from keras.layers import SimpleRNN, Dense

    indim, outdim = data.neurons_required
    return Sequential([
        SimpleRNN(input_dim=indim, output_dim=(500,), activation="tanh"),
        Dense(outdim, activation="sigmoid")
    ]).compile(optimizer="sgd", loss="xent")


def build_network(data: Sequence):
    inshape, outshape = data.neurons_required
    rnn = Network(input_shape=inshape, name="TestRNN")
    rnn.add(RLayer(30, activation="tanh", return_seq=True))
    rnn.add(RLayer(30, activation="tanh"))
    rnn.add(DenseLayer(outshape, activation="sigmoid"))
    rnn.finalize("mse")

    return rnn


def build_LSTM(data: Sequence):
    inshape, outshape = data.neurons_required
    rnn = Network(input_shape = inshape, name="TestLSTM")
    rnn.add(LSTM(20, activation="tanh", return_seq=False))
    rnn.add(DenseLayer(outshape, activation="sigmoid"))
    rnn.finalize("mse")

    return rnn


def xperiment():
    petofi = pull_petofi_data()
    net = build_LSTM(petofi)
    net.describe(verbose=1)
    print("Initial cost: {} acc: {}".format(*net.evaluate(*petofi.table("testing"))))
    print(speak_to_me(net, petofi))

    net.fit(*petofi.table("learning", m=40, shuff=True), epochs=1, verbose=0, shuffle=False)
    if not net.gradient_check(*petofi.table("testing", m=10)):
        return

    X, Y = petofi.table("learning")

    for decade in range(1, 10):
        net.fit(X, Y, 20, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-"*12)
        print("Decade: {0:2<}.5 |".format(decade-1))
        print("-"*12)
        print(speak_to_me(net, petofi))
        net.fit(X, Y, 20, 5, monitor=["acc"], validation=petofi.table("testing"))
        print("-"*12)
        print("Decade: {0:2<} |".format(decade))
        print("-"*12)
        print(speak_to_me(net, petofi))


if __name__ == '__main__':
    xperiment()
