def translate_architecture(arch):
    from ..brainforge.layers import (
        InputLayer, DenseLayer, Activation,
        HighwayLayer, DropOut,
        Flatten, Reshape,
        RLayer, LSTM, EchoLayer,
        PoolLayer, ConvLayer)

    dictionary = {
        "Inpu": InputLayer,
        "Dens": DenseLayer,
        "Acti": Activation,
        "High": HighwayLayer,
        "Drop": DropOut,
        "Flat": Flatten,
        "Resh": Reshape,
        "RLay": RLayer,
        "LSTM": LSTM,
        "Echo": EchoLayer,
        "MaxP": PoolLayer,
        "Conv": ConvLayer
    }

    return dictionary[arch[:4]]
