import numpy as np
import scipy.signal
from time import time


def max_pool(x):
    """Return maximum in groups of 2x2 for a N,h,w image"""
    N, h, w = x.shape
    return np.amax([x[:, (i >> 1) & 1::2, i & 1::2] for i in range(4)], axis=0)


def conv_layer(params, x):
    """Applies a convolutional layer (W,b) followed by 2*2 pool followed by RelU on x"""
    Ws, bs = params
    num_in = Ws.shape[1]
    A = []
    for f, bias in enumerate(bs):
        conv_out = np.sum([scipy.signal.convolve2d(x[i], Ws[f][i], mode='valid') for i in range(num_in)], axis=0)
        A.append(conv_out + bias)
    x = np.array(A)
    x = max_pool(x)
    return np.maximum(x, 0)


def fastconv(params, x):
    Ws, bs = params
    d = x[:, :-1, :-1].swapaxes(0, 1)
    c = x[:, :-1, 1:].swapaxes(0, 1)
    b = x[:, 1:, :-1].swapaxes(0, 1)
    a = x[:, 1:, 1:].swapaxes(0, 1)
    o = Ws[:, :, 0, 0].dot(a)
    o += Ws[:, :, 0, 1].dot(b)
    o += Ws[:, :, 1, 0].dot(c)
    o += Ws[:, :, 1, 1].dot(d)
    o += bs.reshape(-1, 1, 1)
    return o


# 32 filter of depth 16. fX, fY = (2, 2)
Weights = np.random.randn(32, 16, 2, 2).astype(np.float32)
biases = np.random.randn(32).astype(np.float32)
# 25 x 25 pic with depth 16
I = np.random.randn(16, 25, 25).astype(np.float32)

t0 = time()
O = conv_layer((Weights, biases), I)
t1 = time() - t0

t0 = time()
F = fastconv((Weights, biases), I)
t2 = time() - t0

assert np.allclose(O, F), "Not even close!"

print("Slowtime: {}\nFasttime: {}".format(t1, t2))
