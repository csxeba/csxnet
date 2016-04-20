import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.downsample import max_pool_2d

theano.config.exception_verbosity = "high"
# theano.config.optimizer = "fast_compile"

floatX = theano.config.floatX

print("floatX is set to <{}>".format(floatX))


class ConvNetExplicit:
    def __init__(self, data, eta, lmbd,
                 nfilters=1, cfshape=(5, 5), pool=2, hidden1=120, hidden2=60,
                 cost="MSE"):

        assert cfshape[0] == cfshape[1], "Non-sqare filters are not <yet> supported!"

        self.data = data

        channels, x, y = data.learning[0].shape
        cfiltershape = (nfilters, channels, cfshape[0], cfshape[1])
        outputs = len(data.categories)

        cfanin = channels, x, y
        pfanin = (cfanin[1] - cfshape[0] + 1),\
                 (cfanin[2] - cfshape[1] + 1), nfilters
        fcfanin = np.prod((pfanin[0] // pool,
                           pfanin[1] // pool,
                           nfilters))

        inputs = T.tensor4("inputs")
        targets = T.matrix("targets")
        m = T.scalar("m", dtype="int32")
        m_ = m.astype("float32")

        cfilter = theano.shared((np.random.randn(*cfiltershape) /
                                np.sqrt(np.prod(cfanin))).astype(floatX),
                                name="ConvFilters")
        hf1cw = theano.shared((np.random.randn(fcfanin, hidden1) /
                             np.sqrt(fcfanin)).astype(floatX),
                              name="Hidden1Weights")
        hf2cw = theano.shared((np.random.randn(hidden1, hidden2) /
                             np.sqrt(hidden1)).astype(floatX),
                              name="Hidden2Weights")
        outw = theano.shared((np.random.randn(hidden2, outputs) /
                             np.sqrt(hidden2)).astype(floatX),
                             name="OutputWeights")

        cact = nnet.sigmoid(nnet.conv2d(inputs, cfilter))
        pact = T.reshape(
            max_pool_2d(cact, ds=(pool, pool), ignore_border=True),
            # newshape=(m, nfilters * transd_pl[0] * transd_pl[1]))
            newshape=(m, fcfanin))
        fc1act = nnet.sigmoid(pact.dot(hf1cw))
        fc2act = nnet.sigmoid(fc1act.dot(hf2cw))
        outact = nnet.softmax(fc2act.dot(outw))

        # Sqared error cost
        cost = ((targets - outact) ** 2).sum() if cost.lower() == "mse" else \
            nnet.categorical_crossentropy(outact, targets).sum()

        prediction = T.argmax(outact, axis=1)

        l2term = 1 - ((eta * lmbd) / data.N)

        update_cfilter = l2term * cfilter - (eta / m_) * T.grad(cost, cfilter)

        update_hf1cw = l2term * hf1cw - (eta / m_) * T.grad(cost, hf1cw)

        update_hf2cw = l2term * hf2cw - (eta / m_) * T.grad(cost, hf2cw)

        update_outw = l2term * outw - (eta / m_) * T.grad(cost, outw)

        self._train = theano.function(inputs=[inputs, targets, m],
                                      updates=[(cfilter, update_cfilter),
                                               (hf1cw, update_hf1cw),
                                               (hf2cw, update_hf2cw),
                                               (outw, update_outw)],
                                      name="_train")

        self._predict = theano.function(inputs=[inputs, targets, m],
                                        outputs=[cost, prediction],
                                        name="_predict")

    def learn(self, batch_size):
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = questions.shape[0]
            self._train(questions, targets, m)

    def evaluate(self, on="testing"):
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._predict(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc


class ConvNetDynamic:
    def __init__(self, data, eta, lmbd, cost="MSE"):
        self.data = data
        self.fanin, self.outputs = data.neurons_required()
        self.eta = eta
        self.lmbd = lmbd
        self.l2term = 1 - ((eta * lmbd) / data.N)

        self.layers = []
        self.forward_pass_rules = []
        self.update_rules = []

        self.inputs = T.tensor4("Inputs")
        self.targets = T.matrix("Targets")
        self.m = T.scalar("m", dtype="float32")

        self.cost = cost
        self.prediction = None
        self._train = None
        self._predict = None

    def add_convpool(self, conv, filters, pool):
        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape
        self.layers.append(ThConvPoolLayer(conv, filters, pool, fanin, pos))

    def add_fc(self, neurons, activation="sigmoid"):
        if activation == "softmax":
            if neurons != self.outputs:
                print("Warning! Assumed creation of output layer, but got wrong number of neurons!")
                print("Adjusting neuron number to correct output size:", self.outputs)
            self.finalize()

        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape

        self.layers.append(ThFCLayer(neurons, fanin, pos, activation))

    def finalize(self):

        def add_output_layer():
            position = len(self.layers)
            fanin = self.fanin if not position else self.layers[-1].outshape
            self.layers.append(ThOutputLayer(self.outputs, fanin, position))

        def define_feedforward():
            self.forward_pass_rules.append(self.inputs)
            for layer in self.layers:
                self.forward_pass_rules.append(layer.output(self.forward_pass_rules[-1]))

        def define_cost_and_prediction():
            l2normsq = sum([(layer.weights**2).sum() for layer in self.layers])
            self.cost = \
                nnet.categorical_crossentropy(self.forward_pass_rules[-1], self.targets).sum() \
                    if self.cost.lower() == "xent" else \
                ((self.targets - self.forward_pass_rules[-1]) ** 2).sum()
            self.cost = self.cost + (l2normsq * (self.eta * self.lmbd)) / self.data.N

            self.prediction = T.argmax(self.forward_pass_rules[-1], axis=1)

        def define_update_rules():
            for layer in self.layers:
                gradient = (self.eta / self.m) * theano.grad(self.cost, layer.weights)
                self.update_rules.append((
                    layer.weights,
                    layer.weights - gradient
                ))

        add_output_layer()
        define_feedforward()
        define_cost_and_prediction()
        define_update_rules()

        self._train = theano.function(inputs=[self.inputs, self.targets, self.m],
                                      updates=self.update_rules,
                                      name="_train")
        self._predict = theano.function(inputs=[self.inputs, self.targets],
                                        outputs=[self.cost, self.prediction],
                                        name="_predict")

    def learn(self, batch_size):
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = float(questions.shape[0])
            self._train(questions, targets, m)

    def evaluate(self, on="tesing"):
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._predict(qst, idp)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc


class ThConvPoolLayer:
    def __init__(self, conv, filters, pool, inshape, position):
        fanin = np.prod(inshape)
        self.inshape = inshape
        channel, ix, iy = inshape

        assert ix == iy, "Only square convolution is supported!"
        assert ((ix - conv) + 1) % pool == 0, "Non-integer ConvPool output shape!"

        osh = ((ix - conv) + 1) // pool
        self.outshape = osh, osh, filters
        self.position = position

        self.weights = theano.shared(
            (np.random.randn(filters, channel, conv, conv)
             / np.sqrt(fanin)).astype(floatX),
            name="{}. ConvFilters".format(position)
        )

        self.pool = pool

    def output(self, inputs):
        cact = nnet.sigmoid(nnet.conv2d(inputs, self.weights))
        pact = max_pool_2d(cact, ds=(self.pool, self.pool), ignore_border=True)
        return pact


class ThFCLayer:
    def __init__(self, neurons, inshape, position, activation="sigmoid"):
        fanin = np.prod(inshape)
        self.inshape = inshape
        self.outshape = neurons
        self.position = position
        self.activation = {"sigmoid": nnet.sigmoid, "tanh": T.tanh, "softmax": nnet.softmax}[activation.lower()]
        self.weights = theano.shared((np.random.randn(fanin, neurons) / fanin).astype(floatX),
                                     name="{}. FCweights".format(position))

    def output(self, inputs):
        inputs = T.reshape(inputs, (inputs.shape[0], T.prod(inputs.shape[1:])))
        return self.activation(inputs.dot(self.weights))


class ThOutputLayer(ThFCLayer):
    def __init__(self, neurons, inshape, position):
        ThFCLayer.__init__(self, neurons, inshape, position, activation="softmax")


class ThDropoutLayer(ThFCLayer):
    def __init__(self, neurons, inshape, dropchance, position, activation="sigmoid"):
        ThFCLayer.__init__(neurons, inshape, position, activation)
        print("Dropout not implemented yet, falling back to ThFCLayer!")