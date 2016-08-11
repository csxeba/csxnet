"""My crazy expreminets are taking place here"""
from csxnet.brainforge.layers import *
from csxnet.model import NeuralNetworkBase, Network
from csxnet.utilities.nputils import combination

# noinspection PyProtectedMember
from csxnet.brainforge.layers import _LayerBase


class Abomination:
    """
    Experiment #1

    Codename: ABOMINATION

    Description:
    Constructing a neural-level simulated brain and one of its layers
    will be built from complete nerual networks instead of neurons...
    """

    def __init__(self, data, eta, lmbd, cost):
        self.data = data
        self.eta = eta
        self.lmbd = lmbd
        self.cost = cost
        self.layers = []


class AboLayer(_LayerBase):
    def __init__(self, brain: Abomination, no_minions):
        _LayerBase.__init__(self, brain, 0, "sigmoid")
        self.brain = brain
        self.fanin = brain.layers[-1].fanout
        self.neurons = []

        for pos in range(no_minions):
            self.neurons.append(self._forge_minion())

    def _forge_minion(self):
        minion = Network(self.brain.data, self.brain.eta, self.brain.lmbd, 0.0, 0.0, self.brain.cost)
        minion.add_fc(10)
        minion.finalize_architecture()
        return minion

    def feedforward(self, inputs):
        """this ain't so simple after all O.O"""
        pass


class Predictor:
    def __init__(self, brain):
        if isinstance(brain, str):
            brain = wake_ai(brain)
        elif isinstance(brain, NeuralNetworkBase):
            pass
        else:
            raise RuntimeError("Please provide a Network instance for initialization")

        self.params = []
        types = (InputLayer, ConvLayer, PoolLayer, FFLayer, DropOut)
        for layer in brain.layers:
            tp = types.index(type(layer))
            if tp == 1:
                assert layer.fshape[0] == layer.fshape[1] == layer.stride
                self.params.append((layer.stride, ))
            elif tp == 2:
                self.params.append((layer.filter, layer.activation))
            elif tp == 3:
                self.params.append((layer.weights, layer.biases, 1.0, layer.activation))
            elif tp == 4:
                self.params.append((layer.weights, layer.biases, layer.dropchance, layer.activation))
            else:
                pass

    def predict(self, inputs):
        for params in self.params:
            ln = len(params)
            assert ln == 4, "Not implemented yet!"
            inputs = combination(inputs, *params)
        return np.argmax(inputs, axis=1)

    def save(self, path):
        import pickle
        print("Saving predictor instance...")
        with open(path, "wb") as fl:
            pickle.dump(self, fl)
            fl.close()
        print("Predictor saved to {}".format(path))


def wake_ai(path: str):
    import pickle
    import gzip

    with gzip.open(path, "rb") as fl:
        brain = pickle.load(fl)
        fl.close()

    return brain
