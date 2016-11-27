from csxnet.brainforge.layers import RLayer, LSTM, EchoLayer, DropOut, DenseLayer
from csxnet import Network

from csxdata import Sequence, roots


def pull_petofi_data():
    return Sequence(roots["txt"] + "petofi.txt", n_gram=1, timestep=5,
                    cross_val=0.01)


def build_ultimate_recurrent_combo_network(data: Sequence, gradcheck=True):
    inshape, outshape = data.neurons_required
    net = Network(input_shape=inshape, name="TestRNN")
    net.add(LSTM(30, activation="tanh", return_seq=True))
    net.add(RLayer(30, activation="tanh", return_seq=False))
    net.add(EchoLayer(30, activation="tanh"))
    net.add(DenseLayer(80, activation="tanh"))
    if not gradcheck:
        net.add(DropOut(dropchance=0.5))
    net.add(DenseLayer(outshape, activation="sigmoid"))
    net.finalize("xent", optimizer="adam")

    net.fit(*data.table("learning", m=20), batch_size=20, epochs=1, verbose=0)
    if gradcheck:
        net.gradient_check(*data.table("testing", m=20))
    return net


def xperiment():
    petofi = pull_petofi_data()
    model = build_ultimate_recurrent_combo_network(petofi)
    model.fit_csxdata(petofi, monitor=["acc"])


if __name__ == '__main__':
    xperiment()
