from csxnet.nputils import combination

from .NetworkBase import NeuralNetworkBase
from ..Layerdef.Layers import *


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
