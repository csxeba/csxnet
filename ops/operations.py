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
        recfield_size = Fx * Fy * Fc
        ox, oy = tuple(ix - fx + 1 for ix, fx in zip(Xshape, (Fx, Fy)))
        steps = [(sy, sy+ Fy, sx, sx + Fx) for sx in range(ox) for sy in range(oy)]
        rfields = np.zeros((m, len(steps), Fx*Fy*c))

        if Fc != c:
            err = "Supplied filter (F) is incompatible with supplied input! (X)\n"
            err += "input depth: {} != {} :filter depth".format(c, Fc)
            raise ValueError(err)

        # rfields = np.array([[pic[:, sy:ey, sx:ex].ravel() for pic in A]
        #                     for sx, ex, sy, ey in self.calcsteps(Xshape, Fshape)])
        for i, pic in enumerate(A):
            for j, (sy, ey, sx, ex) in enumerate(steps):
                rfields[i][j] = pic[:, sy:ey, sx:ex].ravel()

        output = np.matmul(rfields, F.reshape(recfield_size, nF))
        output = output.transpose(2, 0, 1).reshape(m, nF, ox, oy)
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
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode:", mode)

    def __str__(self):
        return "Convolution"


class ScipySigConv(_OpBase):

    def __init__(self):
        from scipy.signal import correlate
        self.op = correlate

    def __str__(self):
        return "scipy.signal.convolution"

    def __call__(self, A, F, mode="valid"):

        def do_valid():
            out = np.zeros((m, nf, 1, ox, oy))
            for i, pic in enumerate(A):
                for j, filt in enumerate(F):
                    conved = self.op(pic, filt, mode=mode)
                    out[i, j] = conved
            return out[:, :, 0, :, :]

        def do_full():
            out = np.zeros((m, nf, ox, oy))
            for i, pic in enumerate(A):
                for j, filt in enumerate(F):
                    for c in range(fc):
                        out[i, j] += self.op(pic[c], filt[c], mode=mode)
            return out

        m, ic, iy, ix = A.shape
        nf, fc, fy, fx = F.shape
        ox, oy = self.outshape(A.shape, F.shape, mode)

        assert ic == fc, "Number of channels got messed up"

        return do_valid() if mode == "valid" else do_full()

    @staticmethod
    def outshape(inshape, fshape, mode="valid"):
        if mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        elif mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(inshape[-2:], fshape[-2:]))
        else:
            raise RuntimeError("Unsupported mode: " + str(mode))


class MaxPool(_OpBase):

    def __str__(self):
        return "MaxPool"

    def __call__(self, A, fdim):
        m, ch, iy, ix = A.shape
        assert all((iy // fdim == 0, ix // fdim == 0))
        oy, ox = iy // fdim, ix // fdim
        steps = ((sy, sy + fdim, sx, sx + fdim)
                 for sx in range(ox)
                 for sy in range(oy))
        output = np.zeros((m, ch, oy*ox))
        filt = np.zeros_like(A)
        for i, pic in A:
            for c, sheet in enumerate(pic):
                for o, (sy, ey, sx, ex) in enumerate(steps):
                    value = sheet[sy:ey, sx:ex].max()
                    output[i, c, o] = value
                    filt[i, c, sy:ey, sx:ex] = np.equal(output[i, c], value)
        return output.reshape(m, ch, oy, ox), filt

    def outshape(self, inshape, fdim):
        if len(inshape) == 3:
            m, iy, ix = inshape
            return m, iy // fdim, ix // fdim
        elif len(inshape) == 2:
            iy, ix = inshape
            return iy // fdim, ix // fdim




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
