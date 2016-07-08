import numpy as np


def featscale(X: np.ndarray, axis=0, ufctr=(0, 1), dfctr=None, getfctrs=False):
    """Rescales the input by first downscaling between dfctr[0] and dfctr[1], then
    upscaling it between ufctr[0] and ufctr[1]."""
    assert X.ndim == 2, ""
    if dfctr is None:
        dfctr = (X.min(axis=axis), X.max(axis=axis))
    frm, to = ufctr
    output = X - dfctr[0]
    output /= dfctr[1] - dfctr[0]
    output *= (to - frm)
    output += frm

    if not getfctrs:
        return output
    else:
        return output, dfctr, ufctr


def euclidean(itr: np.ndarray, target: np.ndarray):
    """Distance of points in euclidean space"""
    # print("Warning, nputils.euclidean() is untested!")
    return np.sqrt(np.sum(np.square(itr - target), axis=0))


def haversine(coords1: np.ndarray, coords2: np.ndarray):
    """The distance of points on the surface of Earth given their GPS (WGS) coordinates"""
    assert coords1.ndim == coords2.ndim == 2, "Please supply two arrays of coordinate-pairs!"
    assert all([dim1 == dim2 for dim1, dim2 in zip(coords1.shape, coords2.shape)]), \
        "Please supply two arrays of coordinate-pairs!"
    R = 6367  # Approximate radius of Mother Earth in kms
    np.radians(coords1, out=coords1)
    np.radians(coords2, out=coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    e = 2 * np.arcsin(np.sqrt(d))
    return e * R


def ravel_to_matrix(A):
    """Converts an ndarray to a 2d array (matrix) by keeping the first dimension as the rows
    and flattening all the other dimensions to columns"""
    A = np.atleast_2d(A)
    A = A.reshape(A.shape[0], np.prod(A.shape[1:]))
    return A


def logit(Z: np.ndarray):
    """The primitive function of the sigmoid function"""
    return np.log(Z / (1 - Z))


def combination(A, W, b, scale, actfn):
    """Calculates a linear combination, then applies an activation function."""
    return actfn(A.dot(W) + b) * scale


class Test:
    def __init__(self):
        print("\n<<< <<< TESTING |nputils.py| >>> >>>")
        self.featscale()
        self.euclidean()
        self.ravel_to_matrix()
        self.combination()
        print("<<< <<< ALL TEST PASSED @ |nputils.py| >>> >>>\n")

    @staticmethod
    def featscale():
        x = np.arange(3 * 4).reshape((3, 4)).astype(float)
        y = np.array([[0.0, 0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [2.0, 2.0, 2.0, 2.0]])
        output = featscale(x, ufctr=(0, 2))
        assert np.all(np.equal(y, output)), "Feature scale test failed!"
        print("<<< Test @ featscale passed! >>>")

    @staticmethod
    def euclidean():
        def vector():
            x1 = np.zeros((2,)).astype(float)
            x2 = np.ones((2,)).astype(float)
            y = np.sqrt(2)
            output = euclidean(x1, x2)
            assert output == y, "Test failed @ euclidean of vectors!"

        def matrix():
            x1 = np.zeros((2, 2)).astype(float)
            x2 = np.ones((2, 2)).astype(float)
            y = np.sqrt(2) * 2
            output = euclidean(x1, x2).sum()
            assert output == y, "Test failed @ euclidean of matrices!"

        vector()
        matrix()
        print("<<< Test @ euclidean passed! >>>")

    @staticmethod
    def ravel_to_matrix():
        x = np.arange(2*3*4*5*6).reshape((2, 3, 4, 5, 6))
        yshape = (2, 3*4*5*6)
        output = ravel_to_matrix(x).shape
        assert np.all(np.equal(yshape, output)), "Test failed @ ravel_to_matrix!"
        print("<<< Test @ ravel_to_matrix passed! >>>")

    @staticmethod
    def combination():
        def vector_times_scalar():
            x = np.arange(10)
            w = 2
            y = np.arange(0, 20, 2)
            output = combination(x, w, 0.0, 1.0, lambda z: z)
            assert np.all(np.equal(y, output)), "Test failed @ combination of vector with scalar!"

        def vector_times_vector():
            x = np.ones((10,)) * 2
            w = np.arange(10)
            y = float(np.arange(0, 20, 2).sum())
            output = combination(x, w, 0.0, 1.0, lambda z: z)
            assert output == y, "Test failed @ combination of vector with vector!"

        def matrix_times_matrix():
            x = np.arange(12).reshape(3, 4)
            w = np.arange(16).reshape(4, 4)
            y = np.dot(x, w)
            output = combination(x, w, 0.0, 1.0, lambda z: z)
            assert np.all(np.equal(y, output)), "Test failed @ combination of matrix with matrix!"

        vector_times_scalar()
        vector_times_vector()
        matrix_times_matrix()
        print("<<< Test @ combination passed! >>>")


if __name__ == '__main__':
    Test()
