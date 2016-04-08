import numpy as np

import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
from theano.tensor.signal.downsample import max_pool_2d

theano.config.exception_verbosity = "high"


class ConvNet:
    def __init__(self, data, eta, lmbd, nfilters=1, cfshape=(5, 5), pool=2, hidden_fc=120):
        self.data = data
        sliceshape = data.learning[0].shape
        cfiltershape = (nfilters, sliceshape[0], cfshape[0], cfshape[1])
        outputs = len(data.categories)
        l2term = 1 - ((eta * lmbd) / data.N)

        transd_in = sliceshape[-2:]
        transd_cl = [(tri - cfs) + 1 for tri, cfs in zip(transd_in, cfshape)]
        transd_pl = [int(trc / pool) for trc in transd_cl]

        inputs = T.tensor4("inputs", dtype="float64")
        targets = T.matrix("targets", dtype="float64")
        m = T.scalar("m", dtype="int32")

        cfilter = theano.shared(np.random.randn(*cfiltershape) /
                                np.sqrt(np.prod(transd_cl)), name="ConvFilters")
        hfcw = theano.shared(np.random.randn(np.prod(nfilters*np.prod(transd_pl)), hidden_fc) /
                             np.sqrt(np.prod(transd_pl)), name="HiddenFCweights")
        outw = theano.shared(np.random.randn(hidden_fc, outputs) /
                             np.sqrt(hidden_fc), name="OutputFCweights")

        cact = nnet.sigmoid(nnet.conv2d(inputs, cfilter))
        pact = T.reshape(
            max_pool_2d(cact, ds=(pool, pool), ignore_border=True),
            # newshape=(m, nfilters * transd_pl[0] * transd_pl[1]))
            newshape=(m, 784))
        fcact = nnet.sigmoid(pact.dot(hfcw))
        outact = nnet.softmax(fcact.dot(outw))

        # Sqared error cost
        cost = (0.5 * (targets - outact) ** 2).sum()
        # cost = nnet.categorical_crossentropy(outact, targets)

        prediction = T.argmax(outact, axis=1)

        update_cfilter = l2term * cfilter - (eta / m) * T.grad(cost, cfilter)
        update_hfcw = l2term * hfcw - (eta / m) * T.grad(cost, hfcw)
        update_outw = l2term * outw - (eta / m) * T.grad(cost, outw)

        self._train = theano.function(inputs=[inputs, targets, m],
                                      updates=[(cfilter, update_cfilter),
                                               (hfcw, update_hfcw),
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


class FCNet:
    def __init__(self):
        print("Coming soon")
