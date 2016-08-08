import time
import sys

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import sigmoid
from theano.tensor.signal import downsample

from data import CData, mnist_to_lt


datapath = "/data/Prog/data/" if sys.platform.lower() != "win32" else "D:/Data/"
mnistpath = "learning_tables/mnist.pkl.gz"

class Network:
    def __init__(self, data: CData, fshape: tuple, eta: float, lmbd: float):

        l2term = 1 - ((eta*lmbd)/data.N)

        inshape, outshape = data.neurons_required()

        inputs = T.tensor4("X")
        targets = T.matrix("Y")
        m = T.scalar("m", dtype='int32')

        convfilters = theano.shared(np.random.randn(*fshape))
        fcffweights = theano.shared(np.random.randn(5*13*13, outshape))

        convoutput = sigmoid(conv.conv2d(inputs, convfilters))
        pooloutput = T.reshape(downsample.max_pool_2d(convoutput, (2, 2), ignore_border=True), (m, 5*13*13))
        fcffoutput = T.nnet.softmax(pooloutput.dot(fcffweights))

        cost = T.nnet.categorical_crossentropy(fcffoutput, targets).sum()
        # cost = cost + (lmbd * (convfilters**2)).sum() + (lmbd*(fcffweights**2)).sum()
        cost.name = "Regularized Cross-Entropy Cost"

        prediction = T.argmax(fcffoutput, axis=1)

        update_filters = convfilters * T.grad(cost, convfilters)
        update_weights = fcffweights * T.grad(cost, fcffweights)

        # Compile methods
        self.train = theano.function(inputs=[inputs, targets, m],
                                     updates=[(convfilters, update_filters), (fcffweights, update_weights)])
        self.predict = theano.function(inputs=[inputs, targets, m],
                                       outputs=[cost, prediction])

# Set hyperparameters
fshape = (5, 1, 3, 3)
eta = 0.15
lmbd = 5.0

batch_size = 10
epochs = 30

mnist = mnist_to_lt(datapath + mnistpath)

data = CData(mnist, cross_val=1-0.8571428571428571)

net = Network(data, fshape, eta, lmbd)

start = time.time()


# Train the network
for i in range(epochs):
    for batch in data.batchgen(batch_size):
        bsize = batch[0].shape[0]
        net.train(batch[0], batch[1], bsize)
    testtable = data.table(data="testing")
    costval, preds = net.predict(testtable[0], testtable[1], 10000)
    predrate = np.sum(np.equal(preds, data.dummycode("testing"))) / len(preds)
    print("Epoch:\t{}\tCost:\t{}\tAccuracy:\t{}".format(i+1, costval, predrate))

print("Seconds elapsed: {}".format(time.time() - start))
