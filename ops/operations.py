"""Wrappers for vector-operations and other functions"""
import abc

import numpy as np


class _OpBase(abc.ABC):

    @abc.abstractmethod
    def outshape(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Flatten(_OpBase):

    def __init__(self):
        from ..util import rtm
        self.op = rtm

    def __str__(self):
        return "Flatten"

    def __call__(self, A):
        return self.op(A)

    def outshape(self, inshape=None):
        return np.prod(inshape),  # return as tuple!


class Reshape(_OpBase):

    def __init__(self, shape):
        self.shape = shape

    def __str__(self):
        return "Reshape"

    def __call__(self, A):
        m = A.shape[0]
        return A.reshape(m, *self.shape)

    def outshape(self, inshape=None):
        return self.shape


class Convolution(_OpBase):

    def valid(self, A, F):
        m, c, Xshape = A.shape[0], A.shape[1], A.shape[2:]
        nF, Fc, Fy, Fx = F.shape
        Fshape = Fy, Fx
        recfield_size = Fx * Fy * Fc
        if Fc != c:
            err = "Supplied filter (F) is incompatible with supplied input! (X)\n"
            err += "input depth: {} != {} :filter depth".format(c, Fc)
            raise ValueError(err)

        # rfields = np.array([[pic[:, sy:ey, sx:ex].ravel() for pic in A]
        #                     for sx, ex, sy, ey in self.calcsteps(Xshape, Fshape)])
        steps = list(self.calcsteps(Xshape, Fshape))
        rfields = np.zeros((m, len(steps), Fshape[0]*Fshape[1]*c))
        for i, pic in enumerate(A):
            for j, (sy, ey, sx, ex) in enumerate(steps):
                rfields[i][j] = pic[:, sy:ey, sx:ex].ravel()

        oshape = tuple(ix - fx + 1 for ix, fx in zip(Xshape, (Fx, Fy)))
        output = np.matmul(rfields, F.reshape(recfield_size, nF))
        output = output.transpose(2, 0, 1).reshape(m, nF, *oshape)
        return output

    def full(self, A, F):
        fx, fy = F.shape[:2]
        px, py = fx - 1, fy - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (px, px), (py, py)),
                    mode="constant", constant_values=0.)
        return self.valid(pA, F)

    def __call__(self, A, F, mode="valid"):
        return self.valid(A, F) if mode == "valid" else self.full(A, F)

    def outshape(self, inshape, fshape, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[:2]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[:2]))
        else:
            raise RuntimeError("Unsupported mode:", mode)

    def calcsteps(self, inshape, fshape):
        xsteps, ysteps = self.outshape(inshape, fshape)
        fy, fx = fshape

        for sy in range(0, ysteps):
            for sx in range(0, xsteps):
                yield sy, sy + fy, sx, sx + fx

    def __str__(self):
        return "Convolution"


class ScipySigConv:

    def __init__(self):
        from scipy.signal import correlate
        self.op = correlate

    def __str__(self):
        return "scipy.signal.convolution"

    def __call__(self, A, F, mode="valid"):
        m, ic, iy, ix = A.shape
        nf, fc, fy, fx = F.shape
        ox, oy = self.outshape(A.shape, F.shape, mode)

        assert ic == fc, "Number of channels got messed up"

        output = np.zeros((m, nf, 1, ox, oy))
        for i, filt in enumerate(F):
            for j, batch in enumerate(A):
                conved = self.op(batch, filt, mode=mode)
                output[j, i] = conved

        return output[:, :, 0, :, :]

    @staticmethod
    def outshape(inshape=None, fshape=None, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode: " + str(mode))


class _ActivationFunctionBase(_OpBase):

    def __call__(self, Z: np.ndarray): pass

    def __str__(self): raise NotImplementedError

    def derivative(self, Z: np.ndarray):
        raise NotImplementedError

    def backwards(self, A):
        return self.derivative(A)

    @property
    def outshape(self):
        return self._inshape


class _Sigmoid(_ActivationFunctionBase):

    def __call__(self, Z: np.ndarray):
        return np.divide(1.0, np.add(1, np.exp(-Z)))

    def __str__(self): return "sigmoid"

    def derivative(self, A):
        return np.multiply(A, np.subtract(1.0, A))


class _Tanh(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    def derivative(self, A):
        return np.subtract(1.0, np.square(A))


class _Linear(_ActivationFunctionBase):

    def __call__(self, Z):
        return Z

    def __str__(self): return "linear"

    def derivative(self, Z):
        return np.ones_like(Z)


class _ReLU(_ActivationFunctionBase):

    def __call__(self, Z):
        return np.maximum(0.0, Z)

    def __str__(self): return "relu"

    def derivative(self, A):
        d = np.greater(A, 0.0).astype("float32")
        return d


class _SoftMax(_ActivationFunctionBase):

    def __call__(self, Z):
        eZ = np.exp(Z)
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def __str__(self): return "softmax"

    def derivative(self, A: np.ndarray):
        # This is the negative of the outer product of the last axis with itself
        J = A[..., None] * A[:, None, :]  # given by -a_i*a_j, where i =/= j
        iy, ix = np.diag_indices_from(J[0])
        J[:, iy, ix] = A * (1. - A)  # given by a_i(1 - a_j), where i = j
        return J.sum(axis=1)  # sum for each sample


sigmoid = _Sigmoid()
tanh = _Tanh()
linear = _Linear()
relu = _ReLU()
softmax = _SoftMax()

act_fns = {key.lower(): fn for key, fn in locals().items() if key[0] not in ("_", "F")}
