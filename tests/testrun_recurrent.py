from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me
from csxnet import Network

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
    rnn = Network(data, 0.01, 0.0, 0.0, 0.0, cost="xent", name="TestRNN")
    rnn.add_rec(500, activation="tanh")
    rnn.finalize_architecture("sigmoid")

    return rnn


def build_LSTM(data: Sequence):
    rnn = Network(data, 0.01, 0.0, 0.0, 0.0, cost="xent", name="TestLSTM")
    rnn.add_lstm(30, activation="tanh")
    rnn.finalize_architecture("sigmoid")

    return rnn


def xperiment():

    net = build_network(pull_petofi_data())
    net.describe(verbose=1)
    print("Initial cost: {} acc: {}".format(*net.evaluate()))
    print(speak_to_me(net, net.data))
    # if not net.gradient_check():
    #     return

    for decade in range(1, 10):
        net.fit(20, 5, monitor=["acc"])
        print("-"*12)
        print("Decade: {0:2<} |".format(decade))
        print("-"*12)
        print(speak_to_me(net, net.data))
        net.fit(20, 5, monitor=["acc"])
        print("-"*12)
        print("Decade: {0:2<} |".format(decade))
        print("-"*12)
        print(speak_to_me(net, net.data))


if __name__ == '__main__':
    xperiment()
