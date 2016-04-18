import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.downsample import max_pool_2d

theano.config.exception_verbosity = "high"
theano.config.optimizer = "fast_compile"


class ConvNet:
    def __init__(self, data, eta, lmbd, nfilters=1, cfshape=(5, 5), pool=2, hidden_fc=120):

        assert cfshape[0] == cfshape[1], "Non-sqare filters are not <yet> supported!"

        self.data = data
        self.eta = eta
        eta = T.scalar(dtype="float64")
        channels, x, y = data.learning[0].shape
        cfiltershape = (nfilters, channels, cfshape[0], cfshape[1])
        outputs = len(data.categories)
        l2term = 1 - ((eta * lmbd) / data.N)

        cfanin = channels, x, y
        pfanin = (cfanin[1] - cfshape[0] + 1),\
                 (cfanin[2] - cfshape[1] + 1),\
                 nfilters
        fcfanin = np.prod((pfanin[0] // pool,
                           pfanin[1] // pool,
                           nfilters))

        inputs = T.tensor4("inputs", dtype="float64")
        targets = T.matrix("targets", dtype="float64")
        m = T.scalar("m", dtype="int32")

        cfilter = theano.shared(np.random.randn(*cfiltershape) /
                                np.sqrt(np.prod(cfanin)), name="ConvFilters")
        cbiases = theano.shared(np.zeros((nfilters,)), name="ConvBiases")
        hfcw = theano.shared(np.random.randn(fcfanin, hidden_fc) /
                             np.sqrt(fcfanin), name="HiddenFCweights")
        hfcb = theano.shared(np.zeros((hidden_fc,)), name="HiddenFCbiases")
        outw = theano.shared(np.random.randn(hidden_fc, outputs) /
                             np.sqrt(hidden_fc), name="OutputFCweights")
        outb = theano.shared(np.zeros((outputs,)), name="OutputFCbiases")

        cact = nnet.sigmoid(nnet.conv2d(inputs, cfilter))
        pact = T.reshape(
            max_pool_2d(cact, ds=(pool, pool), ignore_border=True),
            # newshape=(m, nfilters * transd_pl[0] * transd_pl[1]))
            newshape=(m, fcfanin))
        fcact = nnet.sigmoid(pact.dot(hfcw))
        outact = nnet.softmax(fcact.dot(outw))

        # Sqared error cost
        cost = ((targets - outact) ** 2).sum()
        # cost = nnet.categorical_crossentropy(outact, targets).sum()

        prediction = T.argmax(outact, axis=1)

        update_cfilter = l2term * cfilter - (eta / m) * T.grad(cost, cfilter)
        # update_cbiases = cbiases - (eta / m) * T.grad(cost, cbiases)

        update_hfcw = l2term * hfcw - (eta / m) * T.grad(cost, hfcw)
        # update_hfcb = hfcb - (eta / m) * T.grad(cost, hfcb)

        update_outw = l2term * outw - (eta / m) * T.grad(cost, outw)
        # update_outb = outb - (eta / m) * T.grad(cost, outb)

        self._train = theano.function(inputs=[inputs, targets, eta, m],
                                      updates=[(cfilter, update_cfilter),
                                               # (cbiases, update_cbiases),
                                               (hfcw, update_hfcw),
                                               # (hfcb, update_hfcb),
                                               (outw, update_outw)],
                                               # (outb, update_outb)],
                                      name="_train")

        self._predict = theano.function(inputs=[inputs, targets, m],
                                        outputs=[cost, prediction],
                                        name="_predict")

    def learn(self, batch_size):
        for i, (questions, targets) in enumerate(self.data.batchgen(batch_size)):
            m = questions.shape[0]
            self._train(questions, targets, self.eta, m)

    def evaluate(self, on="testing"):
        m = self.data.n_testing
        tab = self.data.table(on)
        qst, idp = tab[0][:m], tab[1][:m]
        cost, pred = self._predict(qst, idp, m)
        acc = np.mean(np.equal(np.argmax(idp, axis=1), pred))
        return cost, acc


class FCNet:
    def __init__(self):
        print("Coming soon")
