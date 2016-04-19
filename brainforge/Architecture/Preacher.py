from .NNModel import Network
from ..Layerdef.Layers import *


class Predictor:
    def __init__(self, brain):
        if isinstance(brain, str):
            brain = wake_ai(brain)
        elif isinstance(brain, Network):
            pass
        else:
            raise RuntimeError("Please provide a Network instance for initialization")

        layers = []
        params = []
        types = (InputLayer, ConvLayer, PoolLayer, FFLayer, DropOut)
        for layer in brain.layers:
            tp = types.index(type(layer))
            if tp == 0:
                params.append(1)
            elif tp == 1:
                params.append(layer.filter)
            elif tp == 2:
                assert layer.fshape[0] == layer.fshape[1] == layer.stride
                params.append(layer.stride)
            elif tp == 3:
                params.append((layer.weights, layer.biases))
            elif tp == 4:
                params.append((layer.weights, layer.biases, layer.dropchance))


def wake_ai(source: str):
    import pickle
    import gzip

    fl = gzip.open(source)
    brain = pickle.load(fl)
    fl.close()

    return brain
