import numpy as np


def featscale(X: np.ndarray, frm=0, to=1, axis=0, factors=False):
    fctr = (X.min(axis=axis), X.max(axis=axis))
    X -= fctr[0]
    X /= fctr[1] - fctr[0]
    X *= (to - frm)
    X += frm
    if not factors:
        return X
    else:
        return X, fctr


def euclidean(itr: np.ndarray, target: np.ndarray):
    """Distance of two (or more) vectors in euclidean space"""
    print("Warning, nputils.euclidean() is untested!")
    return np.sqrt(np.square(np.sum(itr - target, axis=0)))


def haversine(coords1: np.ndarray, coords2: np.ndarray):
    """The distance of two points on Earth given their GPS coords"""
    R = 6367  # radius of Mother Earth in kms
    np.radians(coords1, out=coords1)
    np.radians(coords2, out=coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    e = 2 * np.arcsin(np.sqrt(d))
    return e * R


