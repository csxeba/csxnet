import numpy as np

from .utility import outshape, calcsteps


def convolve(matrix, filt, stride):
    msh = tuple(["NaN"] + list(matrix.shape))
    fsh = tuple(["NaN"] + list(filt.shape))
    oshape = outshape(msh, fsh, stride)
    steps = calcsteps(msh, fsh, stride)
    result = np.zeros(len(steps))
    for i, (start0, end0, start1, end1) in enumerate(steps):
        result[i] = frobenius(matrix[start0:end0, start1:end1], filt)
    return result.reshape(oshape)


def frobenius(mat, filt):
    """"Convolve" the filter and the matrix.
    Meaning: compute elementwise (Hadamard) product, then sum everything
    nD Array goes in, scalar comes out."""
    assert mat.shape == filt.shape, "Shapes differ! Can't convolve..."
    return np.sum(mat * filt)


def maxpool(mat):
    return np.amax(mat, axis=(0, 1))


def avgpool(mat):
    return np.average(mat)
