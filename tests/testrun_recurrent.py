from csxdata import Sequence, roots
from csxdata.utilities.helpers import speak_to_me
from csxnet import Network

TIMESTEP = 10
NGRAM = 1


def pull_petofi_data():
    petofi = Sequence(roots["txt"] + "petofi.txt", n_gram=NGRAM, timestep=TIMESTEP,
                      cross_val=0.01)
    return petofi


def build_reference_network(data: Sequence):
    from keras.models import Sequential
    from keras.layers import SimpleRNN, Dense

    indim, outdim = data.neurons_required

    rnn = Sequential([
        SimpleRNN(input_dim=indim, output_dim=(30,), activation="sigmoid"),
        Dense(outdim, activation="sigmoid")
    ])
    rnn.compile(optimizer="sgd", loss="mse")
    return rnn


def build_network(data: Sequence):
    rnn = Network(data, 0.01, 0.0, 0.0, 0.0, cost="mse", name="TestRNN")
    rnn.add_rec(30, activation="sigmoid")
    rnn.finalize_architecture("sigmoid")

    return rnn


def xperiment():

    def perform_gradient_checking():
        from csxnet.util import gradient_check

        net.fit(20, 1, verbose=0)
        gradient_check(net, *net.data.table("testing", m=100), epsilon=1e-4, display=False)

    net = build_network(pull_petofi_data())

    perform_gradient_checking()

    initcost, initacc = net.evaluate()
    print("Initial cost: {} acc: {}".format(initcost, initacc))

    for decade in range(1, 10):
        net.fit(20, 10, monitor=["acc"])
        print("-"*12)
        print("Decade: {0:2<} |".format(decade))
        print("-"*12)
        print(speak_to_me(net, net.data))


if __name__ == '__main__':
    xperiment()
