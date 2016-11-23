"""
REFERENCE CODE FROM:
https://gist.github.com/karpathy/587454dc0146a6ae21fc

All rights go to Andrej Karpathy, the respected owner of the code.
I only Python3-ized it.

ORIGINAL DOCSTRING:
This is a batched LSTM forward and backward pass
"""

import numpy as np


actprime = lambda z: 1. - z**2
sigprime = lambda z: z * (1. - z)


class LSTM:
    @staticmethod
    def init(input_size, hidden_size, fancy_forget_bias_init=3):
        """
        Initialize parameters of the LSTM (both weights and biases in one matrix)
        One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
        """
        # +1 for the biases, which will be the first row of WLSTM
        WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
        WLSTM[0, :] = 0  # initialize biases to zero
        if fancy_forget_bias_init != 0:
            # forget gates get little bit negative bias initially to encourage them to be turned off
            # remember that due to Xavier initialization above, the raw output activations from gates before
            # nonlinearity are zero mean and on order of standard deviation ~1
            WLSTM[0, hidden_size:2 * hidden_size] = fancy_forget_bias_init
        return WLSTM

    @staticmethod
    def forward(X, WLSTM):
        """
        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
        """
        n, b, input_size = X.shape
        d = WLSTM.shape[1] / 4  # hidden size
        # Perform the LSTM forward pass with X as the input
        xphpb = WLSTM.shape[0]  # x plus h plus bias, lol
        Hin = np.zeros((n, b, xphpb))  # input [1, xt, ht-1] to each tick of the LSTM
        Hout = np.zeros((n, b, d))  # hidden representation of the LSTM (gated cell content)
        IFOG = np.zeros((n, b, d * 4))  # input, forget, output, gate (IFOG)
        C = np.zeros((n, b, d))  # cell content
        Ct = np.zeros((n, b, d))  # tanh of cell content
        for t in range(n):
            # concat [x,h] as input to the LSTM
            prevh = Hout[t - 1] if t > 0 else 0.
            Hin[t, :, 0] = 1  # bias
            Hin[t, :, 1:input_size + 1] = X[t]
            Hin[t, :, input_size + 1:] = prevh
            # compute all gate activations. dots: (most work is this line)
            IFOG[t] = Hin[t].dot(WLSTM)
            # non-linearities
            IFOG[t, :, :3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * d]))  # sigmoids; these are the gates
            IFOG[t, :, 3 * d:] = np.tanh(IFOG[t, :, 3 * d:])  # tanh
            I, F, O, cand = np.split(IFOG, 4, axis=-1)
            # compute the cell activation
            prevc = C[t - 1] if t > 0 else 0.
            C[t] = I * cand + F * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = O * Ct[t]

        cache = {'WLSTM': WLSTM,
                 'Hout': Hout,
                 'IFOG': IFOG,
                 'C': C,
                 'Ct': Ct,
                 'Hin': Hin}

        # return C[t], as well so we can continue LSTM with prev state init if needed
        return Hout, C[-1], Hout[-1], cache

    @staticmethod
    def backward(dHout_in, cache, dcn=None, dhn=None):

        WLSTM = cache['WLSTM']
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        n, b, d = Hout.shape
        input_size = WLSTM.shape[0] - d - 1  # -1 due to bias

        # backprop the LSTM
        dIFOG = np.zeros(IFOGf.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n, b, input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        dHout = dHout_in.copy()  # make a copy so we don't have any funny side effects
        if dcn is not None:
            dC[n - 1] += dcn.copy()  # carry over nabla_w from later
        if dhn is not None:
            dHout[n - 1] += dhn.copy()
        for t in reversed(range(n)):

            tanhCt = Ct[t]
            dIFOGf[t, :, 2 * d:3 * d] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1 - tanhCt ** 2) * (IFOGf[t, :, 2 * d:3 * d] * dHout[t])

            if t > 0:
                dIFOGf[t, :, d:2 * d] = C[t - 1] * dC[t]
                dC[t - 1] += IFOGf[t, :, d:2 * d] * dC[t]
            else:
                dIFOGf[t, :, d:2 * d] = 0. * dC[t]
                dc0 = IFOGf[t, :, d:2 * d] * dC[t]
            dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d:] * dC[t]
            dIFOGf[t, :, 3 * d:] = IFOGf[t, :, :d] * dC[t]

            # backprop activation functions
            dIFOG[t, :, 3 * d:] = (1 - IFOGf[t, :, 3 * d:] ** 2) * dIFOGf[t, :, 3 * d:]
            y = IFOGf[t, :, :3 * d]
            dIFOG[t, :, :3 * d] = (y * (1.0 - y)) * dIFOGf[t, :, :3 * d]

            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].T, dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.T)

            # backprop the identity transforms into Hin
            dX[t] = dHin[t, :, 1:input_size + 1]
            if t > 0:
                dHout[t - 1, :] += dHin[t, :, input_size + 1:]
            else:
                dh0 += dHin[t, :, input_size + 1:]

        return dX, dWLSTM, dc0, dh0

    @staticmethod
    def backwardrw(error_above, cache):

        W = cache['WLSTM']
        outputs = cache['Hout']
        gates = cache['IFOG']
        I, F, O, cand = np.split(gates, 4, axis=-1)
        C = cache['C']
        tC = cache['Ct']
        X = cache['Hin']
        time, m, neurons = outputs.shape
        Z = W.shape[0] - neurons - 1  # -1 due to bias

        # backprop the LSTM
        dgates = np.zeros(gates.shape)
        dW = np.zeros(W.shape)
        dZ = np.zeros(X.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((time, m, Z))
        error = error_above.copy()  # make a copy so we don't have any funny side effects

        for t in reversed(range(time)):

            # backprop tanh non-linearity first then continue backprop
            dC[t] += actprime(tC[t]) * (O[t] * error[t])

            if t > 0:
                state_y = C[t - 1]
                dC[t - 1] += dC[t] * F[t]
            else:
                state_y = 0.
            
            dF = state_y * dC[t] 
            dI = cand[t] * dC[t]
            dO = tC[t] * error[t]
            dcand = I[t] * dC[t]

            dgates[t] = np.concatenate((dI, dF, dO, dcand), axis=-1)

            # backprop activation functions
            dgates[t, :, 3 * neurons:] *= actprime(cand[t])
            y = gates[t, :, :3 * neurons]
            dgates[t, :, :3 * neurons] *= sigprime(y)

            # backprop matrix multiply
            dW += np.dot(X[t].T, dgates[t])
            dZ[t] = dgates[t].dot(W.T)

            # backprop the identity transforms into Hin
            dX[t] = dZ[t, :, 1:Z + 1]
            if t > 0:
                error[t - 1, :] += dZ[t, :, Z + 1:]

        return dX, dW

# -------------------
# TEST CASES
# -------------------


def checkSequentialMatchesBatch():
    """ check LSTM I/O forward/backward interactions """

    n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
    input_size = 10
    WLSTM = LSTM.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)

    # sequential forward
    cprev = c0
    hprev = h0
    caches = [{} for _ in range(n)]
    Hcat = np.zeros((n, b, d))
    for t in range(n):
        xt = X[t:t + 1]
        _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
        caches[t] = cache
        Hcat[t] = hprev

    # sanity check: perform batch forward to check that we get the same thing
    H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
    assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

    # eval loss
    wrand = np.random.randn(*Hcat.shape)
    # loss = np.sum(Hcat * wrand)
    dH = wrand

    # get the batched version nabla_w
    BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

    # now perform sequential backward
    dX = np.zeros_like(X)
    dWLSTM = np.zeros_like(WLSTM)
    dc0 = np.zeros_like(c0)
    dh0 = np.zeros_like(h0)
    dcnext = None
    dhnext = None
    for t in reversed(range(n)):
        dht = dH[t].reshape(1, b, d)
        dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
        dhnext = dhprev
        dcnext = dcprev

        dWLSTM += dWLSTMt  # accumulate LSTM gradient
        dX[t] = dx[0]
        if t == 0:
            dc0 = dcprev
            dh0 = dhprev

    # and make sure the nabla_w match
    print()
    'Making sure batched version agrees with sequential version: (should all be True)'
    print()
    np.allclose(BdX, dX)
    print()
    np.allclose(BdWLSTM, dWLSTM)
    print()
    np.allclose(Bdc0, dc0)
    print()
    np.allclose(Bdh0, dh0)


def checkBatchGradient():
    """ check that the batch gradient is correct """

    # lets gradient check this beast
    n, b, d = (5, 3, 4)  # sequence length, batch size, hidden size
    input_size = 10
    WLSTM = LSTM.init(input_size, d)  # input size, hidden size
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)

    # batch forward backward
    H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
    wrand = np.random.randn(*H.shape)
    # loss = np.sum(H * wrand)  # weighted sum is a nice hash to use I think
    dH = wrand
    dX, dWLSTM, dc0, dh0 = LSTM.backwardrw(dH, cache)

    def fwd():
        h, _, _, _ = LSTM.forward(X, WLSTM, c0, h0)
        return np.sum(h * wrand)

    # now gradient check all
    delta = 1e-5
    rel_error_thr_warning = 1e-2
    rel_error_thr_error = 1
    tocheck = [X, WLSTM, c0, h0]
    grads_analytic = [dX, dWLSTM, dc0, dh0]
    names = ['X', 'WLSTM', 'c0', 'h0']
    for j in range(len(tocheck)):
        mat = tocheck[j]
        dmat = grads_analytic[j]
        name = names[j]
        # gradcheck
        for i in range(mat.size):
            old_val = mat.flat[i]
            mat.flat[i] = old_val + delta
            loss0 = fwd()
            mat.flat[i] = old_val - delta
            loss1 = fwd()
            mat.flat[i] = old_val

            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / (2 * delta)

            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0  # both are zero, OK.
                status = 'OK'
            elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
                rel_error = 0  # not enough precision to check this
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
                status = 'OK'
                if rel_error > rel_error_thr_warning:
                    status = 'WARNING'
                if rel_error > rel_error_thr_error:
                    status = '!!!!! NOTOK'

            # print stats
            print('{} checking param {} index {} (val = {}), analytic = {}, numerical = {}, relative error = {}'
                  .format(status, name, np.unravel_index(i, mat.shape), old_val, grad_analytic,
                          grad_numerical, rel_error))


if __name__ == "__main__":
    checkSequentialMatchesBatch()
    input('check OK, press key to continue to gradient check')
    checkBatchGradient()
    print('every line should start with OK. Have a nice day!')
