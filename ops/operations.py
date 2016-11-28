import abc

import numpy as np


class _OpBase(abc.ABC):

    def __init__(self, inshape=None):
        self.inshape = inshape

    @abc.abstractmethod
    def outshape(self, inshape):
        if inshape is None and self.inshape is None:
            raise RuntimeError("Please supply input shape for output shape calculation!")
        return inshape if inshape else self.inshape

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def backwards(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class Flatten(_OpBase):

    def __init__(self):
        from ..util import rtm
        _OpBase.__init__(self)
        self.op = rtm

    def __str__(self):
        return "Flatten"

    def __call__(self, A):
        return self.op(A)

    def backwards(self, A):
        m = A.shape[0]
        return A.reshape(m, *self.inshape)

    def outshape(self, inshape=None):
        ish = _OpBase.outshape(self, inshape)
        return np.prod(ish),  # return as tuple!


class Convolution(_OpBase):

    def __init__(self, filtershape, mode="valid"):
        from ..util import convolve
        _OpBase.__init__(self)
        if mode not in ("valid", "full"):
            raise ValueError("Only valid and full convolution is supported, not {}".format(mode))
        self.mode = mode
        self.op = convolve
        self.filtershape = filtershape

    def valid(self, A, F):
        return self.op(A, F, stride=1)

    def full(self, A, F):
        fx, fy = F.shape[:2]
        px, py = fx - 1, fy - 1
        pA = np.pad(A, pad_width=((0, 0), (0, 0), (px, px), (py, py)),
                    mode="constant", constant_values=0.)
        return self.valid(pA, F)

    def __call__(self, A, F):
        return self.valid(A, F) if self.mode == "valid" else self.full(A, F)

    def backwards(self, E, F):
        rF = np.rot90(F, 2)
        return self.valid(E, rF) if self.mode == "full" else self.full(E, rF)

    def outshape(self, inshape=None, fshape=None):
        ish = _OpBase.outshape(self, inshape)
        assert fshape is not None or self.filtershape is not None
        fsh = fshape if fshape else self.filtershape
        if self.mode == "valid":
            return tuple(ix - fx + 1 for ix, fx in zip(ish[-2:], fsh[:2]))
        elif self.mode == "full":
            return tuple(ix + fx - 1 for ix, fx in zip(ish[-2:], fsh[:2]))

    def __str__(self):
        return "Convolution"


class ScipySigConv(Convolution):

    def __init__(self, filtershape, mode="valid"):
        from scipy.signal import convolve
        Convolution.__init__(self, filtershape, mode)
        self.op = convolve

    def __str__(self):
        return "scipy.signal.convolution"

    def __call__(self, A, F):
        m, ic, ix, iy = A.shape
        fx, fy, fc, nf = F.shape
        ox, oy = self.outshape(A.shape, F.shape)

        assert ic == fc, "Number of channels got messed up"

        F = F.transpose(3, 2, 0, 1)

        output = np.zeros((m, nf, 1, ox, oy))
        for i, filt in enumerate(F):
            for j, batch in enumerate(A):
                conved = self.op(batch, filt, mode=self.mode)
                output[j, i] = conved

        return output[:, :, 0, :, :]

    def backwards(self, E, F):
        self.mode = "full" if self.mode == "valid" else "valid"
        matrix = self(E, np.rot90(F, 2))
        self.mode = "full" if self.mode == "valid" else "valid"
        return matrix


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
