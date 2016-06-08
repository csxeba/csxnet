import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as nnet


floatX = theano.config.floatX


def test():
    net = ThLSTM()
