import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.downsample import max_pool_2d

theano.config.exception_verbosity = "high"
# theano.config.optimizer = "fast_compile"

floatX = theano.config.floatX

print("floatX is set to <{}>".format(floatX))
print("Device used: <{}>".format(theano.config.device))


class ConvNetExplicit:
    def __init__(self, data, eta, lmbd1, lmbd2,
                 nfilters=1, cfshape=(5, 5), pool=2, hidden1=120, hidden2=60,
                 cost="MSE"):

        assert cfshape[0] == cfshape[1], "Non-sqare filters are not <yet> supported!"

        self.data = data
        self.age = 0

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
            newshape=(m, fcfanin))
        fc1act = nnet.sigmoid(pact.dot(hf1cw))
        fc2act = nnet.sigmoid(fc1act.dot(hf2cw))
        outact = nnet.softmax(fc2act.dot(outw))

        l1 = sum(((cfilter**2).sum(), (hf1cw**2).sum(), (hf2cw**2).sum(), (outw**2).sum()))
        l1 *= 0.5 * lmbd1 / (self.data.N * 2)
        l2 = sum((T.abs_(cfilter).sum(), T.abs_(hf1cw).sum(), T.abs_(hf2cw).sum(), T.abs_(outw).sum()))
        l2 *= lmbd2 / (self.data.N * 2)

        cost = T.exp2(targets - outact).sum() if cost.lower() == "mse" else \
            nnet.categorical_crossentropy(outact, targets).sum()
        cost += l1 + l2

        prediction = T.argmax(outact, axis=1)

        update_cfilter = cfilter - (eta / m_) * T.grad(cost, cfilter)

        update_hf1cw = hf1cw - (eta / m_) * T.grad(cost, hf1cw)

        update_hf2cw = hf2cw - (eta / m_) * T.grad(cost, hf2cw)

        update_outw = outw - (eta / m_) * T.grad(cost, outw)

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
        self.age += 1

    def evaluate(self, on="testing"):
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._predict(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc

    def describe(self, verbose=1):
        chain = "---------------\n"
        chain += "Explicit Theano-based Artificial Neural Network.\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture is hardcoded -> please check the source code...\n"
        chain += "---------------"
        if verbose:
            print(chain)
        else:
            return chain


class ConvNetDynamic:
    def __init__(self, data, eta, lmbd1, lmbd2, mu, cost):
        self.data = data
        self.fanin, self.outputs = data.neurons_required()
        self.eta = eta
        self.lmbd1 = lmbd1
        self.lmbd2 = lmbd2
        self.mu = mu

        self.layers = []

        self.inputs = T.tensor4("Inputs")
        self.targets = T.matrix("Targets")
        self.m = T.scalar("m", dtype="float32")
        self.mint = T.scalar("mint", dtype="int32")

        self.cost = cost
        self.output = None
        self.prediction = None
        self._train = None
        self._predict = None

        self.age = 0
        self.architecture = []

    def add_convpool(self, conv, filters, pool):
        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape
        self.architecture.append("{}x{}x{} Conv + {} Pool".format(filters, conv, conv, pool))
        self.layers.append(ThConvPoolLayer(conv, filters, pool, fanin, pos))

    def add_fc(self, neurons, activation="sigmoid"):
        if activation == "softmax":
            if neurons != self.outputs:
                print("Warning! Assumed creation of output layer, but got wrong number of neurons!")
                print("Adjusting neuron number to correct output size:", self.outputs)
            self.finalize()

        pos = len(self.layers)
        fanin = self.fanin if not pos else self.layers[-1].outshape

        self.architecture.append("FC{}: {}".format(neurons, activation[:4]))
        self.layers.append(ThFCLayer(neurons, fanin, pos, activation))

    def finalize(self):

        def define_output_layer():
            position = len(self.layers)
            fanin = self.fanin if not position else self.layers[-1].outshape
            self.architecture.append("Out{}: {}".format(self.outputs, "softmax"))
            return ThOutputLayer(self.outputs, fanin, position)

        def define_feedforward():
            self.mint = self.m.astype("int32")
            forward_pass_rules = [self.inputs]
            for layer in self.layers:
                forward_pass_rules.append(layer.output(forward_pass_rules[-1], self.mint))
            return forward_pass_rules[-1]

        def define_cost_and_prediction():
            l1 = sum([T.abs_(layer.weights).sum() for layer in self.layers])
            l1 *= self.lmbd1 / (self.data.N / 2)
            l2 = sum([T.exp2(layer.weights).sum() for layer in self.layers])
            l2 *= self.lmbd2 / (self.data.N * 2)

            # Build string for architecture display
            chain = "Cost: " + self.cost
            reg = ""
            if self.lmbd1 or self.lmbd2:
                reg += " + "
            if self.lmbd1:
                reg += "L1"
                if self.lmbd2:
                    reg += " + L2 reg."
                else:
                    reg += " reg."
            if self.lmbd2:
                reg += "L2 reg."
            self.architecture.append(chain + reg)

            if self.cost.lower() == "xent":
                cost = nnet.categorical_crossentropy(
                    coding_dist=self.output,
                    true_dist=self.targets).sum()
            elif self.cost.lower() == "mse":
                cost = T.exp2(self.targets - self.output).sum()
            else:
                raise RuntimeError("Cost function {} not supported!".format(self.cost))

            cost += l1 + l2
            prediction = T.argmax(self.output, axis=1)

            return cost, prediction

        def define_update_rules():
            rules = []
            for layer in self.layers:
                rules.append((
                    layer.weights,
                    layer.weights - (self.eta / self.m) * T.grad(self.cost, layer.weights)
                ))
            return rules

        self.layers.append(define_output_layer())
        self.output = define_feedforward()
        self.cost, self.prediction = define_cost_and_prediction()
        update_rules = define_update_rules()

        self._train = theano.function(inputs=[self.inputs, self.targets, self.m],
                                      updates=update_rules,
                                      name="_train")

        self._predict = theano.function(inputs=[self.inputs, self.targets, self.m],
                                        outputs=[self.cost, self.prediction],
                                        name="_predict")

    def learn(self, batch_size):
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = float(questions.shape[0])
            self._train(questions, targets, m)
        self.age += 1

    def evaluate(self, on="tesing"):
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._predict(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc

    def describe(self, verbose=1):
        chain = "---------------\n"
        chain += "Dynamic Theano-based Artificial Neural Network.\n"
        chain += "Age: " + str(self.age) + "\n"
        chain += "Architecture: " + str(self.architecture) + "\n"
        chain += "---------------"
        if verbose:
            print(chain)
        else:
            return chain


class ThConvPoolLayer:
    def __init__(self, conv, filters, pool, inshape, position):
        fanin = np.prod(inshape)
        self.inshape = inshape
        channel, ix, iy = inshape

        assert ix == iy, "Only square convolution is supported!"
        assert ((ix - conv) + 1) % pool == 0, "Non-integer ConvPool output shape!"

        osh = ((ix - conv) + 1) // pool
        self.outshape = osh, osh, filters
        self.fshape = filters, channel, conv, conv
        self.position = position

        self.weights = theano.shared(
            (np.random.randn(*self.fshape)
             / np.sqrt(fanin)).astype(floatX),
            name="{}. ConvFilters".format(position)
        )

        self.pool = pool

    def output(self, inputs, mint):
        cact = nnet.sigmoid(nnet.conv2d(inputs, self.weights))
        return max_pool_2d(cact, ds=(self.pool, self.pool), ignore_border=True)


class ThFCLayer:
    def __init__(self, neurons, inshape, position, activation="sigmoid"):
        self.fanin = np.prod(inshape)
        self.inshape = inshape
        self.outshape = neurons
        self.position = position
        self.activation = {"sigmoid": nnet.sigmoid, "tanh": T.tanh, "softmax": nnet.softmax}[activation.lower()]
        self.weights = theano.shared((np.random.randn(self.fanin, neurons) / self.fanin).astype(floatX),
                                     name="{}. FCweights".format(position))

    def output(self, inputs, mint):
        inputs = T.reshape(inputs, (mint, self.fanin))
        return self.activation(inputs.dot(self.weights))


class ThOutputLayer(ThFCLayer):
    def __init__(self, neurons, inshape, position):
        ThFCLayer.__init__(self, neurons, inshape, position, activation="softmax")


class ThDropoutLayer(ThFCLayer):
    def __init__(self, neurons, inshape, dropchance, position, activation="sigmoid"):
        ThFCLayer.__init__(neurons, inshape, position, activation)
        print("Dropout not implemented yet, falling back to ThFCLayer!")
